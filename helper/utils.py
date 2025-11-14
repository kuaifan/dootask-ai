from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_xai import ChatXAI
from langchain_community.chat_models import (
    ChatZhipuAI,
    ChatTongyi,
    QianfanChatEndpoint,
    ChatCohere
)
from .deepseek import DeepseekChatOpenAI
from .request import RequestClient
from .redis import RedisManager
from .config import THINK_START_PATTERN, THINK_END_PATTERN, REASONING_PATTERN
import os
import time
import json
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def get_model_instance(model_type, model_name, api_key, **kwargs):
    """根据模型类型返回对应的模型实例"""

    base_url = kwargs.get("base_url", None)
    agency = kwargs.get("agency", None)
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 0)
    thinking = kwargs.get("thinking", 0)
    streaming = kwargs.get("streaming", True)

    if model_type == "xai":
        model_type = "grok"

    model_configs = {
        "openai": (ChatOpenAI, {
            "openai_api_key": api_key,
        }),
        "claude": (ChatAnthropic, {
            "anthropic_api_key": api_key,
        }),
        "gemini": (ChatGoogleGenerativeAI, {
            "google_api_key": api_key,
        }),
        "deepseek": (DeepseekChatOpenAI, None),
        "zhipu": (ChatZhipuAI, None),
        "qwen": (ChatTongyi, None),
        "wenxin": (QianfanChatEndpoint, None),
        "cohere": (ChatCohere, None),
        "ollama": (ChatOllama, None),
        "grok": (ChatXAI, None),
    }

    if agency:
        os.environ["https_proxy"] = agency
        os.environ["http_proxy"] = agency

    try:
        model_class, config = model_configs.get(model_type, (None, None))
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        if config is None:
            config = {
                "api_key": api_key,
            }
            if model_type == "wenxin":
                api_key, secret_key = (api_key.split(':') + [None])[:2]
                config.update({"api_key": api_key, "secret_key": secret_key})

        if base_url:
            config.update({"base_url": base_url})

        if max_tokens > 0:
            config.update({"max_tokens": max_tokens})

        if model_type == "openai":
            if thinking > 0:
                config.update({"reasoning_effort": "medium"})
            elif "gpt-5" in model_name.lower() and model_name != "gpt-5-chat-latest":
                config.update({"reasoning_effort": "minimal"})
        else:
            if thinking > 0:
                config.update({"thinking": {"type": "enabled", "budget_tokens": 2000 if thinking == 1 else thinking}})

        common_params = {
            "model": model_name,
            "temperature": temperature,
            "streaming": streaming
        }
        config.update(common_params)
        return model_class(**config)
    except Exception as e:
        raise RuntimeError(f"Failed to create model instance: {str(e)}")
    finally:
        if agency:
            os.environ.pop("https_proxy", None)
            os.environ.pop("http_proxy", None)

def check_timeouts():
    redis_manager = RedisManager()
    while True:
        try:
            # 使用 RedisManager 扫描所有处理中的请求
            current_time = int(time.time())
            for key_id, data in redis_manager.scan_inputs():
                if data and data.get("status") == "prepare":
                    if current_time - data.get("created_at", 0) > 60:
                        # 超时处理
                        data["status"] = "finished"
                        data["response"] = "Request timeout. Please try again."
                        redis_manager.set_input(key_id, data)
                        request_client = RequestClient(
                            server_url=data["server_url"],
                            version=data["version"],
                            token=data["token"],
                            dialog_id=data["dialog_id"]
                        )
                        request_client.call({
                            "update_id": key_id,
                            "update_mark": "no",
                            "text": "Request timeout. Please try again.",
                            "text_type": "md",
                            "silence": "yes"
                        })
        except Exception as e:
            print(f"Error in timeout checker: {str(e)}")
        time.sleep(1)

def get_swagger_ui():
    """Return the Swagger UI HTML content."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Swagger UI</title>
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.1.0/swagger-ui.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.1.0/swagger-ui-bundle.js"></script>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script>
            window.onload = function() {
                SwaggerUIBundle({
                    url: "/swagger.yaml",
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    layout: "BaseLayout"
                });
            }
        </script>
    </body>
    </html>
    """

def filter_end_flag(text: str, flag: str) -> str:
    """
    方案1：使用列表遍历的方式（当前实现）
    时间复杂度：O(n)，其中n是flag的长度
    空间复杂度：O(n)，需要存储所有前缀
    """
    if not text or not flag:
        return text
    
    # 替换完整的flag在全文任何地方
    text = text.replace(flag, '')

    # 生成flag的所有可能的前缀（从完整到单个字符）
    flag_variants = [flag[:i] for i in range(len(flag), 4, -1)]
    
    # 检查文本是否以任何变体结尾
    for variant in flag_variants:
        if text.endswith(variant):
            return text[:-len(variant)].rstrip()
    
    return text.rstrip()

def json_content(content):
    return json.dumps({"content": content}, ensure_ascii=False)

def json_success(success):
    return json.dumps({"success": success}, ensure_ascii=False)

def json_error(error):
    return json.dumps({"error": error}, ensure_ascii=False)

def json_empty():
    return json.dumps({})

def replace_think_content(text):
    # 将 <think>内容</think> 替换为 ::: reasoning 内容 :::
    if '<think' in text or '</think' in text:
        text = THINK_START_PATTERN.sub('::: reasoning\n', text)
        text = THINK_END_PATTERN.sub('\n:::', text)
    return text

def remove_reasoning_content(text):
    # 将 ::: reasoning 内容 ::: 去除
    if "::: reasoning\n" in text:
        text = REASONING_PATTERN.sub('', text)
    return text

def process_html_content(text):
    """
    处理HTML内容，替换图片标签
    :param text: 原始文本
    :return: 处理后的文本
    """
    if not text:
        return text
        
    # 图片计数器
    img_count = 0
    
    # 处理含有alt属性的图片标签
    def replace_img_with_alt(match):
        nonlocal img_count
        img_count += 1
        src = match.group(1) if match.group(1) else ""
        alt = match.group(2) if match.group(2) else ""
        
        if alt:
            return f"[图片{img_count}：{alt}]"
        elif src:
            return f"[图片{img_count}：{src}]"
        else:
            return f"[图片{img_count}]"
    
    # 处理其他图片标签
    def replace_img_without_alt(match):
        nonlocal img_count
        img_count += 1
        return f"[图片{img_count}]"
    
    # 先处理有alt属性的图片
    img_alt_pattern = re.compile(r'<img\s+[^>]*(?:src="([^"]*)")?\s*[^>]*(?:alt="([^"]*)")?\s*[^>]*>', re.IGNORECASE)
    text = img_alt_pattern.sub(replace_img_with_alt, text)
    
    # 处理剩余的图片标签
    img_pattern = re.compile(r'<img\s+[^>]*>', re.IGNORECASE)
    text = img_pattern.sub(replace_img_without_alt, text)
    
    return text

def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]
    
def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)

# 转换为可序列化的字典格式
def message_to_dict(message):
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"type": "system", "content": message.content}
    else:
        raise TypeError("Unknown message type")

# 从字典格式转换为消息对象
def dict_to_message(d):
    """
    将遗留的元组/列表或字典表示转换回LangChain消息。
    """
    msg_type = None
    content = None

    if isinstance(d, dict):
        msg_type = d.get("type") or d.get("role")
        content = d.get("content")
    elif isinstance(d, (list, tuple)) and len(d) >= 2:
        # 旧版本将消息存储为（角色、内容）
        msg_type, content = d[0], d[1]

    if msg_type == "human" or msg_type == "user":
        return HumanMessage(content=content)
    if msg_type == "ai" or msg_type == "assistant":
        return AIMessage(content=content)
    if msg_type == "system":
        return SystemMessage(content=content)

    raise TypeError(f"Unknown message type: {msg_type}")
