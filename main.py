from types import SimpleNamespace
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Form
from fastapi.responses import JSONResponse, StreamingResponse
from helper.utils import get_model_instance, get_swagger_ui, json_empty, json_error, json_content, filter_end_flag, replace_think_content, remove_reasoning_content, process_html_content
from helper.request import Request
from helper.redis import handle_context_limits, RedisManager
from helper.thread_pool import DynamicThreadPoolExecutor
from helper.config import SERVER_PORT, CLEAR_COMMANDS, BYE_WORDS, END_CONVERSATION_MARK, STREAM_TIMEOUT
import json
import time
import random
import string
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
from exceptiongroup import ExceptionGroup
from langchain_core.messages.ai import AIMessage
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import base64
from io import BytesIO
import re


app = FastAPI(
    title="AI Chat API",
    description="基于AI的聊天服务API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Redis 管理器
redis_manager = RedisManager()

# 初始化线程池
thread_pool = DynamicThreadPoolExecutor(
    min_workers=5,
    max_workers=50,
    thread_name_prefix="ai_stream_"
)



# 请求模型
class ChatRequest(BaseModel):
    text: str
    token: str
    dialog_id: int
    dialog_type: Optional[str] = None
    msg_id: Optional[int] = 0
    msg_uid: int
    mention: Optional[int] = 0
    bot_uid: int
    version: str
    extras: Optional[str] = '{}'

class InvokeRequest(BaseModel):
    text: str
    model_type: Optional[str] = 'openai'
    model_name: Optional[str] = 'gpt-3.5-turbo'
    system_message: Optional[str] = None
    api_key: str
    base_url: Optional[str] = None
    agency: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 0
    thinking: Optional[int] = 0
    before_text: Optional[str] = None
    context_key: Optional[str] = None
    context_limit: Optional[int] = 0

class ImageRequest(BaseModel):
    text: str
    model_type: Optional[str] = 'openai'
    model_name: Optional[str] = 'gpt-3.5-turbo'
    system_message: Optional[str] = None
    api_key: str
    base_url: Optional[str] = None
    agency: Optional[str] = None
    image: Optional[UploadFile] = None
    
# 处理聊天请求
@app.post('/chat')
async def chat(request: ChatRequest):
    # 获取参数
    text = request.text
    token = request.token
    dialog_id = request.dialog_id
    dialog_type = request.dialog_type
    msg_id = request.msg_id
    msg_uid = request.msg_uid
    mention = request.mention
    bot_uid = request.bot_uid
    version = request.version
    extras = request.extras
    
    # 检查必要参数是否为空
    if not all([text, token, dialog_id, msg_uid, bot_uid, version]):
        return JSONResponse(content={"code": 400, "error": "Parameter error"}, status_code=200)

    # 解析 extras 参数
    try:
        extras_json = json.loads(extras)
        model_type = extras_json.get('model_type', 'openai')
        model_name = extras_json.get('model_name', 'gpt-3.5-turbo')
        system_message = extras_json.get('system_message')
        server_url = extras_json.get('server_url')
        api_key = extras_json.get('api_key')
        base_url = extras_json.get('base_url')
        agency = extras_json.get('agency')
        temperature = float(extras_json.get('temperature', 0.7))
        max_tokens = int(extras_json.get('max_tokens', 0))
        thinking = int(extras_json.get('thinking', 0))
        before_text = extras_json.get('before_text')
        context_key = extras_json.get('context_key', '')
        context_limit = int(extras_json.get('context_limit', 0))
    except json.JSONDecodeError:
        return JSONResponse(content={"code": 400, "error": "Invalid extras parameter"}, status_code=200)

    # 检查 extras 解析后的必要参数是否为空
    if not all([model_type, model_name, server_url, api_key]):
        return JSONResponse(content={"code": 400, "error": "Parameter error in extras"}, status_code=200)

    # 上下文 before_text 处理
    if not before_text:
        before_text = []
    elif isinstance(before_text, str):
        before_text = [['human', before_text]]
    elif isinstance(before_text, list):
        if before_text and isinstance(before_text[0], str):
            before_text = [['human', text] for text in before_text]

    # 创建请求客户端
    request_client = Request(server_url, version, token, dialog_id)

    # 定义上下文键
    context_key = f"{model_type}_{model_name}_{dialog_id}_{context_key}"

    # 如果是群里的消息
    chat_state_key = ''
    if dialog_type == 'group':
        # 定义对话状态键
        chat_state_key = f"chat_state_{context_key}"

        # 本地判断再见
        if text in BYE_WORDS:
            if redis_manager.get_cache(chat_state_key):
                redis_manager.delete_cache(chat_state_key)
                redis_manager.delete_context(context_key)
                request_client.call({
                    "content": '[{"content":"再见"}]',
                    "silence": "yes",
                }, action='template')   
            return JSONResponse(content={"code": 200, "data": {"desc": "Bye"}}, status_code=200)
        
        # 如果是@消息，开启对话状态
        if mention:
            redis_manager.set_cache(chat_state_key, "active", ex=86400)
        # 如果没有@且不在对话状态中，不回复
        elif not redis_manager.get_cache(chat_state_key):
            return JSONResponse(content={"code": 200, "data": {"desc": "Not in conversation state"}}, status_code=200)
        
        # 添加结束对话提示到系统消息
        end_prompt = f"如果你判断我想要结束对话（比如说再见、谢谢、不打扰了等），请在回复末尾添加标记：{END_CONVERSATION_MARK}。"
        system_message = f"{system_message}\n\n{end_prompt}" if system_message else end_prompt
    
    # 如果是清空上下文的命令
    if text in CLEAR_COMMANDS:
        redis_manager.delete_context(context_key)
        # 调用回调
        request_client.call({
            "notice": "上下文已清空",
            "silence": "yes",
            "source": "ai",
        }, action='notice')
        return JSONResponse(content={"code": 200, "data": {"desc": "Context cleared"}}, status_code=200)

    # 创建消息
    send_id = request_client.call({
        "text": '...',
        "text_type": "md",
        "silence": "yes",
        "reply_id": msg_id,
        "reply_check": "yes",
    })
    if not send_id:
        return JSONResponse(content={"code": 400, "error": "Send message failed"}, status_code=200)

    # 处理HTML内容（图片标签）
    text = process_html_content(text)
    
    # 生成随机8位字符串
    stream_key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    # 将输入存储到 Redis
    redis_manager.set_input(send_id, {
        "text": text,
        "token": token,
        "dialog_id": dialog_id,
        "version": version,

        "before_text": before_text,
        "chat_state_key": chat_state_key,
    
        "model_type": model_type,
        "model_name": model_name,
        "system_message": system_message,
        "server_url": server_url,
        "api_key": api_key,
        "base_url": base_url,
        "agency": agency,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "context_limit": context_limit,

        "context_key": context_key,
        "stream_key": stream_key,
        "created_at": int(time.time()),
        "status": "prepare",
        "response": "",
    })

    # 通知 stream 地址
    request_client.call({
        "userid": msg_uid,
        "stream_url": f"/stream/{send_id}/{stream_key}",
        "source": "ai",
    }, action='stream')

    # 返回成功响应
    return JSONResponse(content={"code": 200, "data": {"id": send_id, "key": stream_key}}, status_code=200)

# 处理流式响应
@app.get('/stream/{msg_id}/{stream_key}')
async def stream(msg_id: str, stream_key: str, token: str = Header(None)):
    if not stream_key:
        async def error_stream():
            yield f"id: {msg_id}\nevent: done\ndata: {json_error('No key')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )

    # 检查 msg_id 是否在 Redis 中
    data = redis_manager.get_input(msg_id)
    if not data:
        async def error_stream():
            yield f"id: {msg_id}\nevent: done\ndata: {json_error('No such ID')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )

    # 检查 stream_key 是否正确
    if stream_key != data["stream_key"]:
        async def error_stream():
            yield f"id: {msg_id}\nevent: done\ndata: {json_error('Invalid key')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )

    # 如果 status 为 finished，直接返回
    if data["status"] == "finished":
        async def finished_stream():
            yield f"""
            id: {msg_id}\nevent: replace\ndata: {json_content(data['response'])}\n\n
            id: {msg_id}\nevent: done\ndata: {json_empty()}\n\n
            """
        return StreamingResponse(
            finished_stream(),
            media_type='text/event-stream'
        )
    client = MultiServerMCPClient(
        {
            "dootask-task": {
                "url": "http://localhost:9000/mcp",
                "transport": "streamable_http",
                "headers": {
                    "token": token
                },
            }
        }
    )
    tools = await client.get_tools()
    async def stream_generate(msg_id, msg_key, data, redis_manager):
        """
        流式生成响应
        """

        is_end = False
        response = ""
        try:
            # 更新数据状态
            data["status"] = "processing"
            redis_manager.set_input(msg_id, data)
            
            # 获取对应的模型实例
            model = get_model_instance(
                model_type=data["model_type"],
                model_name=data["model_name"],
                api_key=data["api_key"],
                base_url=data["base_url"],
                agency=data["agency"],
                temperature=data["temperature"],
                max_tokens=data["max_tokens"],
                thinking=data["thinking"],
                streaming=True,
            )

            # 前置上下文处理
            pre_context = []

            # 添加系统消息到上下文开始
            if data["system_message"]:
                pre_context.append(("system", data["system_message"]))

            # 添加 before_text 到上下文
            if data["before_text"]:
                # 这些模型不支持连续的消息，需要在每条消息之间插入确认消息
                models_need_confirmation = ["deepseek-reasoner", "deepseek-coder"]
                if data["model_name"] in models_need_confirmation:
                    for msg in data["before_text"]:
                        pre_context.append(msg)
                        pre_context.append(["assistant", "好的，明白了。"])
                else:
                    pre_context.extend(data["before_text"])

            # 获取现有上下文
            middle_context = redis_manager.get_context(data["context_key"])

            # 添加用户的新消息
            end_context = [("human", data["text"])]
            
            # 处理模型限制
            final_context = handle_context_limits(
                pre_context=pre_context,
                middle_context=middle_context,
                end_context=end_context,
                model_type=data["model_type"], 
                model_name=data["model_name"], 
                custom_limit=data["context_limit"]
            )

            # 检查上下文是否超限
            if not final_context:
                raise Exception("Context limit exceeded")

            # 缓存配置
            cache_interval = 0.1  # 缓存间隔
            last_cache_time = time.time()

            # 状态变量
            has_reasoning = False
            is_response = False
            
            agent = create_react_agent(model, tools)
            # 开始请求流式响应
            async for chunk, _ in agent.astream(final_context):
                if hasattr(chunk, 'content') and isinstance(chunk.content, list):
                    isContinue = True
                    if chunk.content:
                        chunk = SimpleNamespace(**chunk.content[0])
                        if hasattr(chunk, 'type'):
                            if chunk.type == 'thinking' and hasattr(chunk, 'thinking'):    
                                chunk = SimpleNamespace(reasoning_content=chunk.thinking)
                                isContinue = False
                            elif chunk.type == 'text' and hasattr(chunk, 'text'):
                                chunk = SimpleNamespace(content=chunk.text)
                                isContinue = False
                    if isContinue:
                        continue

                if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content and not is_response:
                    if not has_reasoning:
                        response += "::: reasoning\n"
                        has_reasoning = True
                    response += chunk.reasoning_content
                    response = replace_think_content(response)
                    current_time = time.time()
                    if current_time - last_cache_time >= cache_interval:
                        redis_manager.set_cache(msg_key, filter_end_flag(response, END_CONVERSATION_MARK), ex=STREAM_TIMEOUT)
                        last_cache_time = current_time  
                        
                if hasattr(chunk, 'content') and chunk.content:
                    if has_reasoning:
                        response += "\n:::\n\n"
                        has_reasoning = False
                    is_response = True
                    response += chunk.content
                    response = replace_think_content(response)
                    current_time = time.time()
                    if current_time - last_cache_time >= cache_interval:
                        redis_manager.set_cache(msg_key, filter_end_flag(response, END_CONVERSATION_MARK), ex=STREAM_TIMEOUT)
                        last_cache_time = current_time                    

            # 更新上下文
            if response:    
                redis_manager.extend_contexts(data["context_key"], [
                    ("human", data["text"]),
                    ("assistant", remove_reasoning_content(response))
                ], data["model_type"], data["model_name"], data["context_limit"])

            # 检查是否包含结束对话标记
            if data.get("chat_state_key") and END_CONVERSATION_MARK in response:
                response = filter_end_flag(response, END_CONVERSATION_MARK)
                redis_manager.delete_cache(data["chat_state_key"])
                redis_manager.delete_context(data["context_key"])
                is_end = True            
        except Exception as e:
            # 处理异常
            response = str(e)
        finally:
            # 确保状态总是被更新
            try:
                # 更新完整缓存
                redis_manager.set_cache(msg_key, response, ex=STREAM_TIMEOUT)

                # 更新数据状态
                data["status"] = "finished"
                data["response"] = response
                redis_manager.set_input(msg_id, data)

                # 创建请求客户端
                request_client = Request(
                    server_url=data["server_url"], 
                    version=data["version"], 
                    token=data["token"], 
                    dialog_id=data["dialog_id"]
                )

                # 更新完整消息
                request_client.call({
                    "update_id": msg_id,
                    "update_mark": "no",
                    "text": response,
                    "text_type": "md",
                    "silence": "yes"
                })

                # 如果是结束对话，通知用户
                if is_end:
                    request_client.call({
                        "content": '[{"content":"再见"}]',
                        "silence": "yes",
                    }, action='template')   
            except Exception as e:
                # 记录最终阶段的错误，但不影响主流程
                print(f"Error in cleanup: {str(e)}")

    async def stream_producer():
        """
        流式生产者
        """

        # 生成消息 key
        msg_key = f"stream_msg_{msg_id}"
        
        # 如果是第一个请求，启动异步生产者
        if redis_manager.set_cache(msg_key, "", ex=STREAM_TIMEOUT, nx=True):
            # thread_pool.submit(stream_generate, msg_id, msg_key, data, redis_manager)
            producer_task = asyncio.create_task(stream_generate())

        # 所有请求都作为消费者处理
        wait_start = time.time()
        last_response = ""
        sleep_interval = 0.1  # 睡眠间隔
        timeout_check_interval = 1.0  # 检查超时间隔
        last_timeout_check = time.time()
        check_status_interval = 0.2  # 检查完成状态间隔
        last_status_check = time.time()
        
        while True:
            current_time = time.time()
            
            # 检查超时
            if current_time - last_timeout_check >= timeout_check_interval:
                if current_time - wait_start > STREAM_TIMEOUT:
                    yield f"id: {msg_id}\nevent: replace\ndata: {json_content('Request timeout')}\n\n"
                    yield f"id: {msg_id}\nevent: done\ndata: {json_error('Timeout')}\n\n"
                    if producer_task:
                        producer_task.cancel()
                    return
                last_timeout_check = current_time

            response = redis_manager.get_cache(msg_key)
            if response:
                if not last_response:
                    yield f"id: {msg_id}\nevent: replace\ndata: {json_content(response)}\n\n"
                else:
                    append_response = response[len(last_response):]
                    if append_response:
                        yield f"id: {msg_id}\nevent: append\ndata: {json_content(append_response)}\n\n"
                last_response = response

                # 只在有新响应时才检查状态
                if current_time - last_status_check >= check_status_interval:
                    current_data = redis_manager.get_input(msg_id)
                    if current_data and current_data["status"] == "finished":
                        yield f"id: {msg_id}\nevent: done\ndata: {json_empty()}\n\n"
                        if producer_task:
                            producer_task.cancel()
                        return
                    last_status_check = current_time

            # 睡眠等待
            await asyncio.sleep(sleep_interval)
            # time.sleep(sleep_interval)

    # 返回流式响应
    return StreamingResponse(
        stream_producer(),
        media_type='text/event-stream'
    )

def parse_langchain_response(messages: list) -> dict:
    """
    正确处理LangChain消息对象的解析函数
    """
    result = {
        "final_response": "",
        "metadata": {}
    }
    
    for msg in messages:
        if isinstance(msg, AIMessage):
            print(msg)
            print(msg.response_metadata)
            print(msg.response_metadata["finish_reason"])
        if isinstance(msg, AIMessage) and msg.response_metadata.get("finish_reason") == "stop":
            # print("-------------")
            result["final_response"] = msg.content
            result["metadata"].update({
                "model_name": msg.response_metadata.get("model_name"),
                "prompt_tokens": msg.response_metadata.get("token_usage",{}).get("prompt_tokens"),
                "total_tokens": msg.response_metadata.get("token_usage",{}).get("total_tokens")
            })
    return result

# 处理直接请求
@app.post('/invoke')
async def invoke(request: InvokeRequest, token: str = Header(None)):
    # 获取参数
    text = request.text
    model_type = request.model_type
    model_name = request.model_name
    system_message = request.system_message
    api_key = request.api_key
    base_url = request.base_url
    agency = request.agency
    temperature = request.temperature
    max_tokens = request.max_tokens
    thinking = request.thinking
    before_text = request.before_text
    context_key = request.context_key
    context_limit = request.context_limit
    
    # 检查必要参数是否为空
    if not all([text, api_key]):
        return JSONResponse(content={"code": 400, "error": "Parameter error"}, status_code=200)

    # 上下文 before_text 处理
    if not before_text:
        before_text = []
    elif isinstance(before_text, str):
        before_text = [['human', before_text]]
    elif isinstance(before_text, list):
        if before_text and isinstance(before_text[0], str):
            before_text = [['human', text] for text in before_text]

    # 获取模型实例
    model = get_model_instance(
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        agency=agency,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        streaming=False,
    )

    # 前置上下文
    pre_context = []

    # 添加系统消息到上下文开始
    if system_message:
        pre_context.append(("system", system_message))

    # 添加 before_text 到上下文
    if before_text:
        # 优化处理不同模型的上下文添加逻辑
        models_need_confirmation = {"deepseek-reasoner", "deepseek-coder"}
        for item in before_text:
            pre_context.append(item)
            if model_name in models_need_confirmation:
                pre_context.append(("assistant", "好的，明白了。"))

    # 获取现有上下文
    middle_context = redis_manager.get_context(f"invoke_{context_key}")

    # 添加用户的新消息
    end_context = [("human", text)]

    # 处理模型限制
    final_context = handle_context_limits(
        pre_context = pre_context,
        middle_context = middle_context,
        end_context = end_context,
        model_type=model_type, 
        model_name=model_name, 
        custom_limit=context_limit
    )
    client = MultiServerMCPClient(
        {
            "dootask-task": {
                "url": "http://localhost:9000/mcp",
                "transport": "streamable_http",
                "headers": {
                    "token": token
                },
            }
        }
    )
    # 开始请求直接响应
    try:
        tools = await client.get_tools()
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke(final_context)
        # response = model.invoke(final_context)
        result = parse_langchain_response(response["messages"])
        resContent = replace_think_content(result["final_response"])
        if context_key:
            redis_manager.extend_contexts(f"invoke_{context_key}", [
                ("human", text),
                ("assistant", remove_reasoning_content(resContent))
            ], model_type, model_name, context_limit)
        return JSONResponse(
            content={
                "code": 200, 
                "data": {
                    "content": resContent,
                    "usage": {
                        "total_tokens": response.usage_metadata.get("total_tokens", 0),
                        "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                        "completion_tokens": response.usage_metadata.get("output_tokens", 0)
                    }
                }
            },
            status_code=200
        )
    except ExceptionGroup as eg:
        # 处理异常组中的每个异常
        for exc in eg.exceptions:
            if isinstance(exc, httpx.ConnectError):
                raise HTTPException(
                    status_code=503,
                    detail="后端服务不可用"
                )
        raise HTTPException(
            status_code=500,
            detail="处理请求时发生多个错误"
        )
    except Exception as e:
        return JSONResponse(content={"code": 500, "error": str(e)}, status_code=200)
# 健康检查
@app.get('/health')
async def health():
    try:
        # 检查 Redis 连接
        redis_manager = RedisManager()
        redis_manager.client.ping()
        return JSONResponse(content={"status": "healthy", "redis": "connected"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)


def encode_image(image_file):
    """将图片转换为base64编码"""
    buffered = BytesIO()
    image_file.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_image_message(image_base64):
    """创建包含图片的消息"""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

def parse_result(result):
    """解析结果"""
    if not result:
        return {"error": "解析失败"}
    match = re.search(r'\{[\s\S]*\}', result)
    if match:
        try:
            data = match.group().replace("\n", "")
            return json.loads(data)
        except Exception as e:
            return {"error": "转换json失败"}
    else:
        return {"error": "匹配失败"}

@app.post("/analyze_image")
async def analyze_image(
    image: UploadFile = File(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
    api_key: str = Form(...),
    base_url: str = Form(...),
    agency: str = Form(...)
):
    try:
        model = get_model_instance(
            model_type=model_type,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            agency=agency,
            streaming=False,
        )
        # 读取上传的图片
        contents = await image.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        
        # 将图片转换为base64
        # image_base64 = encode_image(image_data)
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的饭店菜单助手。"),
            ("human", [{"type": "text", "text": "整理菜单信息,将结果以json格式返回"}, create_image_message(image_base64)])
        ])
        
        # 创建LCEL链
        chain = (
            {"image": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        
        # 执行分析
        result = chain.invoke(image_base64)
        print(result)
        data = parse_result(result)
        return {
            'success': True,
            'data': data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Swagger UI route
@app.get('/swagger')
async def swagger():
    return get_swagger_ui()

# Swagger YAML route
@app.get('/swagger.yaml')
async def swagger_yaml():
    return app.send_static_file('swagger.yaml')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=SERVER_PORT)
