# 历史图片处理设计方案

## 背景

前端发送的上下文中可能包含多条带图片的消息。为了节省 token 同时保留 AI 查看历史图片的能力，需要在 dootask-ai 层做处理。

## 需求概述

1. **处理历史消息中的图片**：将历史消息中的图片替换为占位符，缓存原始数据
2. **新增 MCP 工具获取历史图片**：AI 需要时可通过工具获取图片进行分析

## 设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 端点范围 | 两者都需要 | `/invoke/stream` 和 `/stream` |
| 缓存过期时间 | 2 小时 | 平衡内存占用和对话时长 |
| 工具返回格式 | 多模态格式 | AI 可直接分析图片 |
| 工具实现方式 | 内置工具 | 无需额外进程，可访问本地 Redis |

## 整体架构

```
前端请求 (带多模态上下文)
    ↓
┌─────────────────────────────────────┐
│  图片预处理层 (新增)                  │
│  - 识别最后一条 human 消息            │
│  - 历史图片 → 计算 MD5 → 存入 Redis   │
│  - 替换为占位符 [Picture:history_xxx] │
└─────────────────────────────────────┘
    ↓
处理后的上下文 → LLM
    ↓
AI 需要查看历史图片时
    ↓
┌─────────────────────────────────────┐
│  get_history_image 工具 (新增)        │
│  - 输入: image_md5                   │
│  - 从 Redis 读取图片                  │
│  - 返回多模态 content block          │
└─────────────────────────────────────┘
```

## 文件变更

### 新增文件

| 文件 | 职责 |
|------|------|
| `helper/history_image.py` | 图片预处理 + 缓存管理 |
| `helper/tools.py` | 内置工具定义（get_history_image） |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `helper/mcp.py` | 加载内置工具，与 MCP 工具合并 |
| `main.py` | 在两个端点调用图片预处理 |

## 详细设计

### 1. 图片预处理逻辑 (`helper/history_image.py`)

**核心函数：** `process_history_images(messages: List) -> List`

```python
async def process_history_images(messages):
    # 1. 找到最后一条 human 消息的索引
    last_human_idx = find_last_human_index(messages)

    # 2. 遍历所有消息
    for i, msg in enumerate(messages):
        if msg.role != "human":
            continue
        if i == last_human_idx:
            continue  # 跳过最后一条，保留完整图片

        # 3. 处理该消息中的图片
        msg.content = await replace_images_with_placeholders(msg.content)

    return messages
```

**图片替换逻辑：**

```python
async def replace_images_with_placeholders(content):
    # content 可能是 str 或 List[dict]
    if isinstance(content, str):
        return content

    new_content = []
    for item in content:
        if item.get("type") == "image_url":
            # 提取 base64 数据
            url = item["image_url"]["url"]
            if not url.startswith("data:"):
                new_content.append(item)
                continue

            base64_data, mime_type = extract_base64_and_mime(url)

            # 计算 MD5
            md5_hash = hashlib.md5(base64_data.encode()).hexdigest()

            # 存入 Redis (key: history_image_{md5}, TTL: 2小时)
            cache_value = {"data": base64_data, "mime_type": mime_type}
            await redis.set_cache(f"history_image_{md5_hash}", json.dumps(cache_value), ex=7200)

            # 替换为占位符文本
            new_content.append({
                "type": "text",
                "text": f"[Picture:history_{md5_hash}]"
            })
        else:
            new_content.append(item)

    return new_content
```

**Redis Key 格式：** `dootask_ai:cache:history_image_{md5_hash}`

### 2. 内置工具设计 (`helper/tools.py`)

```python
from langchain_core.tools import BaseTool

class GetHistoryImageTool(BaseTool):
    name: str = "get_history_image"
    description: str = """获取历史对话中用户上传的图片。
当用户询问历史图片的细节（如"刚才那张图的右上角是什么"）时，
使用此工具获取图片内容进行分析。
输入参数为图片的 MD5 哈希值（从 [Picture:history_xxx] 占位符中提取）。"""

    response_format: str = "content_and_artifact"

    async def _arun(self, image_md5: str) -> tuple:
        # 1. 参数验证
        if not image_md5 or len(image_md5) < 8:
            return ([{"type": "text", "text": "无效的图片标识符"}], None)

        # 2. 规范化 key（支持带或不带 history_ 前缀）
        if image_md5.startswith("history_"):
            md5_hash = image_md5[8:]
        else:
            md5_hash = image_md5

        # 3. 从 Redis 读取
        cache_key = f"history_image_{md5_hash}"
        cached = await redis.get_cache(cache_key)

        # 4. 未找到则返回错误
        if not cached:
            return ([{"type": "text", "text": "图片不存在或已过期"}], None)

        # 5. 解析缓存数据
        cache_data = json.loads(cached)
        base64_data = cache_data["data"]
        mime_type = cache_data.get("mime_type", "image/jpeg")

        # 6. 返回多模态内容
        return ([{
            "type": "image",
            "mime_type": mime_type,
            "base64": base64_data
        }], None)
```

### 3. 工具加载 (`helper/mcp.py`)

```python
async def load_mcp_tools_for_model(...) -> List:
    # 原有逻辑：加载 MCP 工具
    mcp_tools = await client.get_tools()

    # 新增：加载内置工具
    from helper.tools import load_builtin_tools
    builtin_tools = load_builtin_tools()

    # 合并返回
    all_tools = mcp_tools + builtin_tools
    return [_wrap_tool_with_error_handling(t) for t in all_tools]
```

### 4. 端点集成 (`main.py`)

**直连模式 `/invoke/stream/{stream_key}`：**

```python
async def invoke_stream(stream_key: str, ...):
    # 获取输入数据
    input_data = await redis_manager.get_input(stream_key)
    context = input_data.get("context", [])

    # 新增：处理历史图片
    from helper.history_image import process_history_images
    processed_context = await process_history_images(context)

    # 继续原有流程...
    final_context = await process_vision_content(processed_context, model_name)
```

**对话模式 `/stream/{msg_id}/{stream_key}`：**

```python
async def stream_response(msg_id: str, stream_key: str, ...):
    # 获取上下文
    context = await redis_manager.get_context(context_key)

    # 新增：处理历史图片
    from helper.history_image import process_history_images
    processed_context = await process_history_images(context)

    # 继续原有流程...
```

**处理顺序：**
```
原始上下文
    ↓
process_history_images()  ← 新增：替换历史图片为占位符
    ↓
process_vision_content()  ← 现有：处理当前图片（压缩等）
    ↓
发送给 LLM
```

## 边界情况处理

| 场景 | 处理方式 |
|------|---------|
| 上下文中无图片 | 直接返回原始消息，不做处理 |
| 只有一条 human 消息且含图片 | 保留完整图片，不替换 |
| 相同图片多次出现 | MD5 相同，Redis 覆盖写入（幂等） |
| 图片已是占位符格式 | 跳过，不重复处理 |
| 非 base64 图片（HTTP URL） | 跳过，仅处理 `data:` 开头的 base64 |

## 错误处理

**图片预处理时：**
```python
try:
    md5_hash = hashlib.md5(base64_data.encode()).hexdigest()
    await redis.set_cache(...)
except Exception as e:
    logger.warning(f"Failed to cache history image: {e}")
    # 降级：保留原始图片，不替换
    new_content.append(item)
```

**工具获取图片时：**
```python
async def _arun(self, image_md5: str) -> tuple:
    if not image_md5 or len(image_md5) < 8:
        return ([{"type": "text", "text": "无效的图片标识符"}], None)

    base64_data = await redis.get_cache(cache_key)
    if not base64_data:
        return ([{"type": "text", "text": "图片不存在或已过期"}], None)
```

## 数据流示例

**前端发送的 context：**
```json
[
  ["human", [{"type":"image_url", "image_url":{"url":"data:..."}}, {"type":"text", "text":"这是什么？"}]],
  ["assistant", "这是一只猫..."],
  ["human", "右上角那个东西是什么？"],
  ["assistant", "右上角是一个花瓶..."],
  ["human", [{"type":"image_url", "image_url":{"url":"data:..."}}, {"type":"text", "text":"这张新图呢？"}]]
]
```

**dootask-ai 处理后发给模型：**
```json
[
  ["human", "[Picture:history_abc123] 这是什么？"],
  ["assistant", "这是一只猫..."],
  ["human", "右上角那个东西是什么？"],
  ["assistant", "右上角是一个花瓶..."],
  ["human", [{"type":"image_url", "image_url":{"url":"data:..."}}, {"type":"text", "text":"这张新图呢？"}]]
]
```

**AI 需要查看历史图片时：**
```
→ 调用 get_history_image("abc123")
→ 返回 [{"type": "image", "mime_type": "image/jpeg", "base64": "..."}]
```
