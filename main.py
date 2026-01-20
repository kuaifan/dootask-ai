# 标准库导入
import asyncio
import json
import logging
import random
import string
import time
from pathlib import Path
from types import SimpleNamespace

# 第三方库导入
from exceptiongroup import ExceptionGroup
from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk

# 本地模块导入
from helper.config import SERVER_PORT, CLEAR_COMMANDS, STREAM_TIMEOUT, UI_DIST_PATH
from helper.invoke import parse_context, build_invoke_stream_key
from helper.lifespan import lifespan_context
from helper.models import (
    ModelListError,
    get_models_list,
)
from helper.context import handle_context_limits
from helper.request import RequestClient
from helper.utils import (
    convert_message_content_to_string,
    dict_to_message,
    get_model_instance,
    get_swagger_ui,
    json_empty,
    json_error,
    json_content,
    message_to_dict,
    remove_tool_calls,
    replace_think_content,
    remove_reasoning_content,
    clean_messages_for_ai,
    process_html_content,
    get_reasoning_content
)
from helper.mcp import (
    MCPConfigError,
    load_mcp_config_data,
    save_mcp_config_data,
    load_mcp_tools_for_model,
)
from helper.vision import (
    VisionConfigError,
    load_vision_config,
    save_vision_config,
    process_vision_content,
)
from helper.session_image import process_session_images
from helper.config import VISION_DATA_DIR

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger("ai")

# 工具函数
def ui_assets_available() -> bool:
    """检查 UI 资源是否可用"""
    return UI_DIST_PATH.exists() and UI_DIST_PATH.is_dir()

# FastAPI 应用初始化
app = FastAPI(
    title="AI Chat API",
    description="基于AI的聊天服务API",
    version="1.0.0",
    lifespan=lifespan_context
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/chat", methods=["GET", "POST"])
async def chat(request: Request):
    """接收 chat 请求，校验参数并准备 SSE 数据流所需的缓存信息。"""
    # 智能参数提取
    if request.method == "GET":
        params = dict(request.query_params)
    else:
        form_data = await request.form()
        params = dict(form_data)

    # 参数配置
    defaults = {
        'dialog_id': 0,
        'msg_id': 0,
        'msg_uid': 0,
        'mention': 0,
        'bot_uid': 0,
        'extras': '{}'
    }
    
    # 应用默认值和类型转换
    for key, default_value in defaults.items():
        value = params.get(key, default_value)
        if isinstance(default_value, int):
            try:
                params[key] = int(value)
            except (ValueError, TypeError):
                params[key] = default_value
        else:
            params[key] = value
    
    text = params.get("text")
    token = params.get("token")
    version = params.get("version")
    dialog_id, msg_id, msg_uid, mention, bot_uid, extras = (
        params[k] for k in ["dialog_id", "msg_id", "msg_uid", "mention", "bot_uid", "extras"]
    )

    # 检查必要参数是否为空
    if not all([text, token, dialog_id, msg_uid, bot_uid, version]):
        return JSONResponse(content={"code": 400, "error": "Parameter error"}, status_code=200)

    # 解析 extras 参数
    try:
        extras_json = json.loads(extras)
        model_type = extras_json.get('model_type', 'openai')
        model_name = extras_json.get('model_name', 'gpt-5-nano')
        system_message = extras_json.get('system_message')
        server_url = extras_json.get('server_url')
        api_key = extras_json.get('api_key')
        base_url = extras_json.get('base_url')
        agency = extras_json.get('agency')
        temperature = float(extras_json.get('temperature', 0.7))
        max_tokens = int(extras_json.get('max_tokens', 0))
        thinking = int(extras_json.get('thinking', 0))
        before_text = extras_json.get('before_text')
        before_clear = int(extras_json.get('before_clear', 0))
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
        before_text = [HumanMessage(content=before_text)]
    elif isinstance(before_text, list):
        if before_text and isinstance(before_text[0], str):
            before_text = [HumanMessage(content=text) for text in before_text]

    # 创建请求客户端
    request_client = RequestClient(server_url, version, token, dialog_id)

    # 定义上下文键
    context_key = f"{model_type}_{model_name}_{dialog_id}_{context_key}"
    
    # 如果是清空上下文的命令
    if text in CLEAR_COMMANDS:
        await app.state.redis_manager.delete_context(context_key)
        # 调用回调
        asyncio.ensure_future(request_client.call({
            "notice": "上下文已清空",
            "silence": "yes",
            "source": "ai",
        }, action='notice'))
        return JSONResponse(content={"code": 200, "data": {"desc": "Context cleared"}}, status_code=200)

    # 如果需要在请求前清空上下文
    if before_clear:
        await app.state.redis_manager.delete_context(context_key)

    # 创建消息
    send_id = await request_client.call({
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

    # 处理多模态内容（图片）
    if isinstance(text, list):
        # Multimodal content - process images
        vision_config = load_vision_config()
        text = await process_vision_content(text, model_name, vision_config)

    # 生成随机8位字符串
    stream_key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    # 将输入存储到 Redis
    await app.state.redis_manager.set_input(send_id, {
        "text": text,
        "token": token,
        "dialog_id": dialog_id,
        "version": version,
        "msg_user_token": params.get("msg_user[token]"),
        "before_text": before_text,
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
    asyncio.create_task(request_client.call({
        "userid": msg_uid,
        "stream_url": f"/stream/{send_id}/{stream_key}",
        "source": "ai",
    }, action='stream'))

    # 返回成功响应
    return JSONResponse(content={"code": 200, "data": {"id": send_id, "key": stream_key}}, status_code=200)

@app.get('/stream/{msg_id}/{stream_key}')
async def stream(msg_id: str, stream_key: str):
    """校验 msg_id/stream_key 并通过 SSE 发送缓存的对话输出。"""
    if not stream_key:
        async def error_stream():
            yield f"id: {msg_id}\nevent: done\ndata: {json_error('No key')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )

    # 检查 msg_id 是否在 Redis 中
    data = await app.state.redis_manager.get_input(msg_id)

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
            yield f"id: {msg_id}\nevent: replace\ndata: {json_content(data['response'])}\n\n"
            yield f"id: {msg_id}\nevent: done\ndata: {json_empty()}\n\n"
        return StreamingResponse(
            finished_stream(),
            media_type='text/event-stream'
        )

    dootask_available = bool(getattr(app.state, "dootask_mcp", False))
    tools = await load_mcp_tools_for_model(
        data.get("model_name", ""),
        dootask_available=dootask_available,
        token_candidates=[data.get("msg_user_token"), data.get("token")],
        redis_manager=app.state.redis_manager,
    )
    async def stream_generate(msg_id, msg_key, data, redis_manager):
        """
        流式生成响应
        """

        response = ""
        try:
            # 更新数据状态
            data["status"] = "processing"
            await redis_manager.set_input(msg_id, data)
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
                pre_context.append(SystemMessage(content=data["system_message"]))

            # 添加 before_text 到上下文
            if data["before_text"]:
                # 这些模型不支持连续的消息，需要在每条消息之间插入确认消息
                models_need_confirmation = ["deepseek-reasoner", "deepseek-coder"]
                if data["model_name"] in models_need_confirmation:
                    for msg in data["before_text"]:
                        pre_context.append(msg)
                        pre_context.append(AIMessage(content="好的，明白了。"))
                else:
                    pre_context.extend(data["before_text"])

            # 获取现有上下文
            middle_context = await redis_manager.get_context(data["context_key"])

            # 添加 MCP 工具使用提示
            hint_cache_key = None
            if dootask_available:
                hint_cache_key = f"mcp_hint_shown_{data['context_key']}"
                # 检查是否需要添加提示：1) Redis 无标记 2) pre_context 中无该提示
                hint_cache_value = await redis_manager.get_cache(hint_cache_key)
                has_hint_in_context = any(isinstance(msg, SystemMessage) and "get_message_list" in msg.content for msg in pre_context)
                if not hint_cache_value and not has_hint_in_context:
                    hint_content = (
                        f"如果用户的提问涉及历史对话内容或需要查看完整聊天记录，"
                        f"请使用 get_message_list 工具获取（dialog_id: {data['dialog_id']}）。"
                    )
                    # 尝试追加到现有 SystemMessage，否则插入新的
                    for index, msg in enumerate(pre_context):
                        if isinstance(msg, SystemMessage):
                            pre_context[index] = SystemMessage(content=f"{msg.content}\n\n{hint_content}")
                            break
                    else:
                        pre_context.insert(0, SystemMessage(content=hint_content))

            middle_messages = []
            if middle_context:
                middle_messages = [dict_to_message(msg_dict) for msg_dict in middle_context]

            # 处理历史图片（替换中间上下文中的图片为占位符）
            try:
                middle_dicts = [message_to_dict(m) for m in middle_messages]
                processed_middle = await process_session_images(middle_dicts, redis_manager)
                middle_messages = [dict_to_message(d) for d in processed_middle]
            except Exception as e:
                logger.warning("Failed to process session images in stream: %s", e)

            # 添加用户的新消息
            end_context = [HumanMessage(content=data["text"])]
            # 处理模型限制
            final_context = handle_context_limits(
                pre_context=pre_context,
                middle_context=middle_messages,
                end_context=end_context,
                model_type=data["model_type"],
                model_name=data["model_name"],
                custom_limit=data["context_limit"],
                default_ratio=0.9
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
            # 记录是否使用了 get_message_list 工具
            has_used_get_message_list = False
            # 记录已显示的工具调用（避免重复显示）
            displayed_tool_calls = set()

            agent = create_agent(model, tools)

            # 清理消息中的工具调用标记，避免干扰 AI
            clean_context = clean_messages_for_ai(final_context)

            # 开始请求流式响应
            async for chunk in agent.astream({"messages": clean_context}, stream_mode="messages"):
                # logger.info(chunk)
                msg, metadata = chunk
                if "skip_stream" in metadata.get("tags", []):
                    continue

                # 检测 MCP 工具调用并追加到响应
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name')
                        tool_id = tool_call.get('id')
                        if tool_id and tool_id not in displayed_tool_calls:
                            displayed_tool_calls.add(tool_id)
                            if has_reasoning:
                                response += "\n:::\n\n"
                                has_reasoning = False
                            response += f"\n> <tool-use>Tool: {tool_name}</tool-use>\n\n"
                            await redis_manager.set_cache(msg_key, response, ex=STREAM_TIMEOUT)
                            if tool_name == 'get_message_list':
                                has_used_get_message_list = True

                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                if not isinstance(msg, AIMessageChunk):
                    continue

                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    isContinue = True
                    if msg.content:
                        chunk = SimpleNamespace(**msg.content[0])
                        if hasattr(chunk, 'type'):
                            if chunk.type == 'thinking' and hasattr(chunk, 'thinking'):    
                                chunk = SimpleNamespace(reasoning_content=chunk.thinking)
                                isContinue = False
                            elif chunk.type == 'reasoning' and hasattr(chunk, 'reasoning'):    
                                chunk = SimpleNamespace(reasoning_content=chunk.reasoning)
                                isContinue = False
                            elif chunk.type == 'text' and hasattr(chunk, 'text'):
                                chunk = SimpleNamespace(content=chunk.text)
                                isContinue = False
                    if isContinue:
                        continue

                reasoning_content = get_reasoning_content(msg)
                if reasoning_content and not is_response:
                    if not has_reasoning:
                        response += "::: reasoning\n"
                        has_reasoning = True
                    response += convert_message_content_to_string(reasoning_content)
                    response = replace_think_content(response)
                    current_time = time.time()
                    if current_time - last_cache_time >= cache_interval:
                        await redis_manager.set_cache(msg_key, response, ex=STREAM_TIMEOUT)
                        last_cache_time = current_time  
                        
                if hasattr(msg, 'content') and msg.content:
                    if has_reasoning:
                        response += "\n:::\n\n"
                        has_reasoning = False
                    is_response = True
                    response += convert_message_content_to_string(remove_tool_calls(msg.content))
                    response = replace_think_content(response)
                    current_time = time.time()
                    if current_time - last_cache_time >= cache_interval:
                        await redis_manager.set_cache(msg_key, response, ex=STREAM_TIMEOUT)
                        last_cache_time = current_time                    

            # 更新上下文
            if response:
                await redis_manager.extend_contexts(data["context_key"], [
                    message_to_dict(HumanMessage(content=data["text"])),
                    message_to_dict(AIMessage(content=remove_reasoning_content(response)))
                ], data["model_type"], data["model_name"], data["context_limit"])

            # 如果本次使用了 get_message_list 工具，标记不再需要提示
            if hint_cache_key and has_used_get_message_list:
                await redis_manager.set_cache(hint_cache_key, "1", ex=2592000)

        except Exception as e:
            # 处理异常
            logger.exception(e)
            response = str(e)
        finally:
            # 确保状态总是被更新
            try:
                # 更新完整缓存
                await redis_manager.set_cache(msg_key, response, ex=STREAM_TIMEOUT)

                # 更新数据状态
                data["status"] = "finished"
                data["response"] = response
                await redis_manager.set_input(msg_id, data)

                # 创建请求客户端
                request_client = RequestClient(
                    server_url=data["server_url"], 
                    version=data["version"], 
                    token=data["token"], 
                    dialog_id=data["dialog_id"]
                )

                # 更新完整消息
                asyncio.ensure_future(request_client.call({
                    "update_id": msg_id,
                    "update_mark": "no",
                    "text": response,
                    "text_type": "md",
                    "silence": "yes"
                }))
            except Exception as e:
                # 记录最终阶段的错误，但不影响主流程
                logger.error(f"Error in cleanup: {str(e)}")

    async def stream_producer():
        """
        流式生产者
        """

        # 生成消息 key
        msg_key = f"stream_msg_{msg_id}"
        producer_task = None
        
        # 如果是第一个请求，启动异步生产者
        if await app.state.redis_manager.set_cache(msg_key, "", ex=STREAM_TIMEOUT, nx=True):
            producer_task = asyncio.create_task(stream_generate(msg_id, msg_key, data, app.state.redis_manager))

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

            response = await app.state.redis_manager.get_cache(msg_key)
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
                    current_data = await app.state.redis_manager.get_input(msg_id)
                    if current_data and current_data["status"] == "finished":
                        yield f"id: {msg_id}\nevent: done\ndata: {json_empty()}\n\n"
                        if producer_task:
                            producer_task.cancel()
                        return
                    last_status_check = current_time

            # 睡眠等待
            await asyncio.sleep(sleep_interval)

    # 返回流式响应
    return StreamingResponse(
        stream_producer(),
        media_type='text/event-stream'
    )

@app.post('/invoke/auth')
@app.get('/invoke/auth')
async def invoke_auth(request: Request, token: str = Header(..., alias="Authorization")):
    """
    创建直连流请求并返回可供 SSE 连接的 stream_key。
    """
    if request.method == "GET":
        params = dict(request.query_params)
    else:
        form_data = await request.form()
        params = dict(form_data)
    defaults = {
        'model_type': 'openai',
        'model_name': 'gpt-5-chat',
        'max_tokens': 0,
        'temperature': 0.7,
        'thinking': 0,
    }
    
    # 应用默认值和类型转换
    for key, default_value in defaults.items():
        value = params.get(key, default_value)
        if isinstance(default_value, int):
            try:
                params[key] = int(value)
            except (ValueError, TypeError):
                params[key] = default_value
        else:
            params[key] = value
    
    context_messages = parse_context(params.get("context"))
    api_key = params.get('api_key')
    base_url = params.get('base_url')
    agency = params.get('agency')
    try:
        context_limit = int(params.get('context_limit', 0))
    except (ValueError, TypeError):
        context_limit = 0

    model_type, model_name, max_tokens, temperature, thinking = (
        params[k] for k in defaults.keys()
    )

    # 检查必要参数是否为空
    if not all([context_messages, api_key]):
        return JSONResponse(content={"code": 400, "error": "Parameter error"}, status_code=200)
    
    stream_key = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    storage_key = build_invoke_stream_key(stream_key)

    await app.state.redis_manager.set_input(storage_key, {
        "final_context": [message_to_dict(content) for content in context_messages],
        "model_type": model_type,
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "user_token": token,
        "agency": agency,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "context_limit": context_limit,
        "status": "pending",
        "response": "",
        "created_at": int(time.time()),
    })

    return JSONResponse(
        content={
        "code": 200, 
        "data": {
            "stream_key": stream_key,
            "stream_url": f"/invoke/stream/{stream_key}"
        }
    })

@app.post('/invoke/stream/{stream_key}')
@app.get('/invoke/stream/{stream_key}')
async def invoke_stream(request: Request, stream_key: str):
    """使用 stream_key 返回直连模型的 SSE 响应流。"""
    if not stream_key:
        async def error_stream():
            yield f"id: 0\nevent: done\ndata: {json_error('No key')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )
    storage_key = build_invoke_stream_key(stream_key)
    # 检查 msg_id 是否在 Redis 中
    data = await app.state.redis_manager.get_input(storage_key)
    if not data:
        async def error_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error('No such ID')}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type='text/event-stream'
        )


    # 如果 status 为 finished，直接返回
    if data["status"] == "finished" and data.get("response"):
        async def finished_stream():
            yield f"id: {stream_key}\nevent: replace\ndata: {json_content(data['response'])}\n\n"
            yield f"id: {stream_key}\nevent: done\ndata: {json_empty()}\n\n"
        return StreamingResponse(
            finished_stream(),
            media_type='text/event-stream'
        )    
    
    if data.get("status") == "processing":
        async def processing_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error('Stream is processing')}\n\n"
        return StreamingResponse(
            processing_stream(),
            media_type='text/event-stream'
        )    

    stored_context = data.get("final_context") or []
    parsed_context = parse_context(stored_context)
    if not parsed_context:
        async def no_context_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error('No context found')}\n\n"
        return StreamingResponse(
            no_context_stream(),
            media_type='text/event-stream'
        )

    # 处理历史图片（替换为占位符，缓存到 Redis）
    try:
        context_as_dicts = [message_to_dict(msg) for msg in parsed_context]
        processed_dicts = await process_session_images(
            context_as_dicts,
            app.state.redis_manager
        )
        parsed_context = [dict_to_message(d) for d in processed_dicts]
    except Exception as e:
        logger.warning("Failed to process session images, using original context: %s", e)

    # 处理上下文中的图片内容
    vision_config = load_vision_config()
    model_name = data.get("model_name", "")
    processed_context = []
    for msg in parsed_context:
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            # Multimodal message - process images
            processed_content = await process_vision_content(msg.content, model_name, vision_config)
            processed_context.append(type(msg)(content=processed_content))
        else:
            processed_context.append(msg)
    parsed_context = processed_context

    # 处理上下文限制：拆分为 pre_context（系统提示）、middle_context（历史）、end_context（当前输入）
    # 优先级：end_context > pre_context > middle_context
    pre_context = []
    middle_context = []
    end_context = parsed_context[-1:] if parsed_context else []

    # 检查第一条消息是否是系统提示词
    remaining = parsed_context[:-1] if len(parsed_context) > 1 else []
    if remaining and isinstance(remaining[0], SystemMessage):
        pre_context = [remaining[0]]
        middle_context = remaining[1:]
    else:
        middle_context = remaining

    # 应用 token 限制
    final_context = handle_context_limits(
        pre_context=pre_context,
        middle_context=middle_context,
        end_context=end_context,
        model_type=data.get("model_type"),
        model_name=data.get("model_name"),
        custom_limit=data.get("context_limit", 0),
        default_ratio=0.9,
    )

    if not final_context:
        async def context_exceeded_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error('Context limit exceeded')}\n\n"
        return StreamingResponse(
            context_exceeded_stream(),
            media_type='text/event-stream'
        )

    try:

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
        tools = await load_mcp_tools_for_model(
            data.get("model_name", ""),
            dootask_available=bool(getattr(app.state, "dootask_mcp", False)),
            token_candidates=[data.get("user_token"), data.get("token")],
            redis_manager=app.state.redis_manager,
        )
        agent = create_agent(model, tools)

    except Exception as exc:
        async def model_error_stream():
            yield f"id: {stream_key}\nevent: done\ndata: {json_error(str(exc))}\n\n"
        return StreamingResponse(
            model_error_stream(),
            media_type='text/event-stream'
        )

    async def stream_invoke_response():
        response_text = ""
        last_sent = ""
        has_reasoning = False
        is_response = False
        displayed_tool_calls = set()
        data["status"] = "processing"
        await app.state.redis_manager.set_input(storage_key, data)
        try:
            # 清理消息中的工具调用标记，避免干扰 AI
            clean_context = clean_messages_for_ai(final_context)

            async for chunk in agent.astream({"messages": clean_context}, stream_mode="messages"):
                # logger.info(chunk)
                msg, metadata = chunk
                if "skip_stream" in metadata.get("tags", []):
                    continue

                # 检测 MCP 工具调用并追加到响应
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name')
                        tool_id = tool_call.get('id')
                        if tool_id and tool_id not in displayed_tool_calls:
                            displayed_tool_calls.add(tool_id)
                            if has_reasoning:
                                response_text += "\n:::\n\n"
                                has_reasoning = False
                            response_text += f"\n> <tool-use>Tool: {tool_name}</tool-use>\n\n"

                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                if not isinstance(msg, AIMessageChunk):
                    continue

                if hasattr(chunk, "content") and isinstance(msg.content, list):
                    should_continue = True
                    if msg.content:
                        chunk = SimpleNamespace(**msg.content[0])
                        if hasattr(chunk, "type"):
                            if chunk.type == "thinking" and hasattr(chunk, "thinking"):
                                chunk = SimpleNamespace(reasoning_content=chunk.thinking)
                                should_continue = False
                            elif chunk.type == "reasoning" and hasattr(chunk, "reasoning"):
                                chunk = SimpleNamespace(reasoning_content=chunk.reasoning)
                                should_continue = False
                            elif chunk.type == "text" and hasattr(chunk, "text"):
                                chunk = SimpleNamespace(content=chunk.text)
                                should_continue = False
                    if should_continue:
                        continue

                reasoning_content = get_reasoning_content(msg)
                if reasoning_content and not is_response:
                    if not has_reasoning:
                        response_text += "::: reasoning\n"
                        has_reasoning = True
                    response_text += reasoning_content
                    response_text = replace_think_content(response_text)
                    
                if hasattr(msg, "content") and msg.content:
                    if has_reasoning:
                        response_text += "\n:::\n\n"
                        has_reasoning = False
                    is_response = True
                    response_text += msg.content
                    response_text = replace_think_content(response_text)

                if response_text != last_sent:
                    if last_sent and response_text.startswith(last_sent):
                        delta = response_text[len(last_sent):]
                        event_type = "append"
                    else:
                        delta = response_text
                        event_type = "replace"
                    if delta:
                        yield f"id: {stream_key}\nevent: {event_type}\ndata: {json_content(delta)}\n\n"
                    last_sent = response_text

            data["status"] = "finished"
            data["response"] = response_text
            await app.state.redis_manager.set_input(storage_key, data)
            yield f"id: {stream_key}\nevent: done\ndata: {json_empty()}\n\n"
        except Exception as exc:
            data["status"] = "finished"
            data["response"] = response_text or str(exc)
            data["error"] = str(exc)
            await app.state.redis_manager.set_input(storage_key, data)
            yield f"id: {stream_key}\nevent: done\ndata: {json_error(str(exc))}\n\n"
    return StreamingResponse(
        stream_invoke_response(),
        media_type='text/event-stream'
    )

@app.get('/')
async def root():
    """返回 UI 页面的入口或简单的健康提示。"""
    if not ui_assets_available():
        return JSONResponse(content={"message": "DooTask AI service"}, status_code=200)
    return FileResponse(UI_DIST_PATH / 'index.html')

@app.get('/ui/')
@app.get('/ui/{path:path}')
async def ui_assets(path: str = 'index.html'):
    """提供静态 UI 文件，若不存在则返回默认 index.html。"""
    if not ui_assets_available():
        return JSONResponse(content={"error": "UI assets not available"}, status_code=404)

    safe_path = path.lstrip("/")
    target = UI_DIST_PATH / safe_path
    if target.exists() and target.is_file():
        return FileResponse(target)

    return FileResponse(UI_DIST_PATH / 'index.html')

@app.get('/models/list')
async def models_list(type: str = '', base_url: str = '', key: str = '', agency: str = ''):
    """返回可以用于前端展示的模型列表。"""
    model_type = type.strip()
    base_url = base_url.strip()
    key = key.strip()
    agency = agency.strip()

    try:
        data = get_models_list(
            model_type,
            base_url=base_url or None,
            key=key or None,
            agency=agency or None,
        )
    except ModelListError as exc:
        return JSONResponse(content={"code": 400, "error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - defensive logging
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Failed to get model list")
        return JSONResponse(content={"code": 500, "error": "获取失败"}, status_code=500)

    return JSONResponse(content={"code": 200, "data": data}, status_code=200)

@app.get('/mcp/config')
async def get_mcp_config():
    """获取MCP配置列表"""
    try:
        data = load_mcp_config_data()
        return JSONResponse(content={"code": 200, "data": data})
    except MCPConfigError as exc:
        logger.error(f"Failed to read MCP config: {exc}")
        return JSONResponse(content={"code": 500, "error": "Failed to read MCP config"}, status_code=500)

@app.post('/mcp/config')
async def save_mcp_config(request: Request):
    """保存MCP配置列表"""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse(content={"code": 400, "error": "Invalid MCP config payload"}, status_code=400)
        save_mcp_config_data(data)
        return JSONResponse(content={"code": 200, "data": {"message": "ok"}})
    except MCPConfigError as exc:
        logger.error(f"Failed to save MCP config: {exc}")
        return JSONResponse(content={"code": 500, "error": "Failed to save MCP config"}, status_code=500)

@app.get('/vision/config')
async def get_vision_config():
    """获取视觉识别配置"""
    try:
        config = load_vision_config()
        return JSONResponse(content={"code": 200, "data": config})
    except Exception as exc:
        logger.error(f"Failed to get vision config: {exc}")
        return JSONResponse(content={"code": 500, "error": "Failed to read vision config"}, status_code=500)

@app.post('/vision/config')
async def save_vision_config_endpoint(request: Request):
    """保存视觉识别配置"""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse(content={"code": 400, "error": "Invalid config format"}, status_code=400)
        save_vision_config(data)
        return JSONResponse(content={"code": 200, "data": {"message": "ok"}})
    except VisionConfigError as exc:
        logger.error(f"Failed to save vision config: {exc}")
        return JSONResponse(content={"code": 500, "error": "Failed to save vision config"}, status_code=500)

@app.get('/vision/preview/{filename}')
async def vision_preview(filename: str):
    """图片预览接口"""
    # Security: only allow specific extensions and no path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(content={"code": 400, "error": "Invalid filename"}, status_code=400)

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        return JSONResponse(content={"code": 400, "error": "Invalid file type"}, status_code=400)

    filepath = VISION_DATA_DIR / filename
    if not filepath.exists():
        return JSONResponse(content={"code": 404, "error": "File not found"}, status_code=404)

    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")

    return FileResponse(filepath, media_type=media_type)

@app.get('/health')
async def health():
    """执行 Redis 等资源的简单健康检查。"""
    try:

        await app.state.redis_manager.client.ping()
        return JSONResponse(content={"status": "healthy", "redis": "connected"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)

@app.get('/swagger')
async def swagger():
    """返回内置的 Swagger UI 页面。"""
    return get_swagger_ui()

@app.get('/swagger.yaml')
async def swagger_yaml():
    """提供 Swagger YAML 描述文件。"""
    static_file_path = Path(__file__).resolve().parent / "static" / "swagger.yaml"
    return FileResponse(static_file_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8080,reload=True)
