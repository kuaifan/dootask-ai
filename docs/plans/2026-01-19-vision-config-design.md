# 视觉识别配置功能设计

## 概述

为 DooTask AI 添加视觉识别配置功能，支持用户配置哪些模型可以接收图片，并在接口层面支持多模态消息格式。

## 需求总结

- **配置定位**：全局开关
- **配置内容**：启用开关、支持的模型列表、图片处理参数
- **界面入口**：和 MCP 配置平级的卡片
- **配置存储**：单独文件 `config/vision-config.json`
- **接口格式**：遵循 OpenAI/LangChain 多模态格式

---

## 一、数据结构

### 1.1 视觉配置文件

**路径**: `config/vision-config.json`

```json
{
  "enabled": true,
  "supportedModels": [
    {"id": "gpt-4o", "name": "GPT-4o"},
    {"id": "claude-opus-4-5", "name": "Claude Opus 4.5"}
  ],
  "maxImageSize": 2048,
  "maxFileSize": 10,
  "compressionQuality": 80
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| enabled | boolean | 是否启用视觉识别功能 |
| supportedModels | array | 允许接收图片的模型列表 |
| maxImageSize | number | 最大图片尺寸（像素，宽高中较大值） |
| maxFileSize | number | 最大文件大小（MB） |
| compressionQuality | number | 压缩质量（1-100） |

### 1.2 模型定义扩展

**文件**: `helper/config.py`

在 `DEFAULT_MODELS` 中为每个模型添加 `support_vision` 字段：

```python
{"id": "gpt-4o", "name": "GPT-4o", "support_mcp": True, "support_vision": True},
{"id": "gpt-4o-mini", "name": "GPT-4o Mini", "support_mcp": True, "support_vision": True},
{"id": "o1", "name": "o1", "support_mcp": False, "support_vision": False},
```

---

## 二、后端逻辑

### 2.1 新增视觉配置模块

**文件**: `helper/vision.py`

```python
# 核心函数
load_vision_config()                    # 加载视觉配置
save_vision_config(data)                # 保存视觉配置
is_vision_enabled()                     # 检查是否启用
model_supports_vision(model_name)       # 检查模型是否支持视觉
process_vision_content(content, config) # 处理图片内容（压缩/缩放/保存）
```

### 2.2 图片处理流程

```
接收请求 (text/content 包含图片)
    |
检查视觉功能是否启用 且 模型在 supportedModels 中
    |
  +-- 是 -----------------------------+    +-- 否 -----------------------------+
  | 直接使用视觉能力                   |    | 降级为 URL + MCP OCR              |
  | - 解码 base64                     |    | - 解码 base64                     |
  | - 检查/压缩/缩放                  |    | - 保存到本地文件                  |
  | - 构建多模态 HumanMessage         |    | - 生成文件名 (如 uuid.jpg)        |
  |   content=[text, image_url]       |    | - 替换为 URL 文本描述             |
  +-----------------------------------+    |   "[图片: http://nginx/ai/        |
                                           |    vision/preview/uuid.jpg]"      |
                                           | - AI 可通过 MCP OCR 识别          |
                                           +-----------------------------------+
```

### 2.3 图片存储与清理

| 项目 | 说明 |
|------|------|
| 存储路径 | `data/vision/` 目录 |
| 文件命名 | `{uuid}.{ext}` |
| 访问接口 | `GET /vision/preview/{filename}` |
| AI 访问 URL | `http://nginx/ai/vision/preview/{filename}` |
| 清理策略 | 定期任务，删除 7 天前的文件 |

### 2.4 API 接口

```python
GET  /vision/config              # 获取视觉配置
POST /vision/config              # 保存视觉配置
GET  /vision/preview/{filename}  # 图片预览接口
```

### 2.5 定期清理任务

在 `helper/lifespan.py` 中添加定时任务：
- 启动时执行一次清理
- 之后每 24 小时执行一次
- 删除 `data/vision/` 中超过 7 天的文件

---

## 三、前端界面

### 3.1 新增组件

| 文件 | 说明 |
|------|------|
| `ui/src/components/aibot/VisionConfigCard.tsx` | 视觉配置卡片（和 MCPListCard 平级） |
| `ui/src/components/aibot/VisionEditorSheet.tsx` | 视觉配置编辑弹窗 |
| `ui/src/lib/vision-storage.ts` | 配置加载/保存接口封装 |
| `ui/src/data/vision-config.ts` | TypeScript 类型定义 |

### 3.2 卡片展示

```
+------------------------------------------+
|  视觉识别                          [编辑] |
|  ---------------------------------------- |
|  状态: 已启用                             |
|  支持模型: GPT-4o, Claude Opus 4.5 +2     |
|  图片限制: 2048px / 10MB / 80%            |
+------------------------------------------+
```

### 3.3 编辑弹窗

```
+------------------------------------------+
|  视觉识别配置                        [x]  |
+------------------------------------------+
|  启用视觉识别              [开关]         |
|                                          |
|  支持的模型                              |
|  +-- openai ----------------------------+|
|  | [ ] GPT-4o  [ ] GPT-4o Mini  [ ] o1  ||
|  +--------------------------------------+|
|  +-- claude ----------------------------+|
|  | [ ] Claude Opus 4.5  [ ] Sonnet 4    ||
|  +--------------------------------------+|
|  ...                                     |
|                                          |
|  最大图片尺寸 (px)    [    2048    ]     |
|  最大文件大小 (MB)    [      10    ]     |
|  压缩质量 (1-100)     [      80    ]     |
|                                          |
|               [取消]  [保存]             |
+------------------------------------------+
```

### 3.4 国际化

在 `ui/src/lib/i18n-core.ts` 添加 `vision` 命名空间：

| Key | 中文 | 英文 |
|-----|------|------|
| title | 视觉识别 | Vision Recognition |
| enabled | 启用视觉识别 | Enable Vision |
| supportedModels | 支持的模型 | Supported Models |
| maxImageSize | 最大图片尺寸 | Max Image Size |
| maxFileSize | 最大文件大小 | Max File Size |
| compressionQuality | 压缩质量 | Compression Quality |
| statusEnabled | 已启用 | Enabled |
| statusDisabled | 已禁用 | Disabled |

---

## 四、接口数据格式

### 4.1 `/chat` 接口 - text 参数

**纯文本（现有，保持兼容）：**
```json
{ "text": "你好" }
```

**多模态格式（新增）：**
```json
{
  "text": [
    {"type": "text", "text": "请分析这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]
}
```

### 4.2 `/invoke/auth` 接口 - context 参数

**纯文本消息（现有，保持兼容）：**
```json
{
  "context": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

**多模态消息（新增）：**
```json
{
  "context": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请分析这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

### 4.3 处理逻辑位置

| 接口 | 处理位置 | 说明 |
|------|----------|------|
| `/chat` | `main.py` 中 `process_html_content` 之后 | 处理 text 参数中的图片 |
| `/invoke/auth` | `helper/invoke.py` 的 `_normalize_message` | 支持多模态 content 格式 |
| `/invoke/stream` | stream 处理前 | 对 final_context 中的图片进行处理 |

---

## 五、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `helper/config.py` | 修改 | DEFAULT_MODELS 添加 support_vision |
| `helper/vision.py` | **新增** | 视觉配置与图片处理模块 |
| `helper/invoke.py` | 修改 | `_normalize_message` 支持多模态 content |
| `helper/lifespan.py` | 修改 | 添加图片清理定时任务 |
| `main.py` | 修改 | 添加视觉 API，`/chat` 和 `/invoke/stream` 集成图片处理 |
| `config/vision-config.json` | **新增** | 默认视觉配置文件 |
| `data/vision/` | **新增** | 图片存储目录 |
| `ui/src/components/aibot/VisionConfigCard.tsx` | **新增** | 配置卡片组件 |
| `ui/src/components/aibot/VisionEditorSheet.tsx` | **新增** | 编辑弹窗组件 |
| `ui/src/lib/vision-storage.ts` | **新增** | 前端存储接口 |
| `ui/src/data/vision-config.ts` | **新增** | TypeScript 类型 |
| `ui/src/lib/i18n-core.ts` | 修改 | 添加国际化文本 |
| `ui/src/App.tsx` | 修改 | 集成视觉配置卡片 |
| `ui/src/data/aibots.ts` | 修改 | 模型添加 support_vision 字段 |

---

## 六、实现顺序建议

1. **后端基础**
   - 创建 `helper/vision.py` 模块
   - 修改 `helper/config.py` 添加 support_vision 字段
   - 创建默认配置文件 `config/vision-config.json`

2. **后端 API**
   - 添加 `/vision/config` GET/POST 接口
   - 添加 `/vision/preview/{filename}` 图片预览接口

3. **图片处理逻辑**
   - 实现 `process_vision_content` 函数
   - 修改 `/chat` 接口集成图片处理
   - 修改 `helper/invoke.py` 支持多模态 content
   - 修改 `/invoke/stream` 集成图片处理

4. **定时任务**
   - 在 `helper/lifespan.py` 添加图片清理任务

5. **前端界面**
   - 创建 TypeScript 类型定义
   - 创建存储接口封装
   - 创建 VisionConfigCard 组件
   - 创建 VisionEditorSheet 组件
   - 集成到 App.tsx
   - 添加国际化文本
   - 修改 aibots.ts 添加 support_vision 字段
