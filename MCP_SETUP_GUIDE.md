# MCP 配置指南

本指南介绍如何在 DooTask AI 中配置和使用 Model Context Protocol (MCP)。

## 功能概述

MCP 配置功能允许管理员为 AI 助手添加额外的功能扩展，通过配置 MCP 服务器来增强 AI 的能力。

## 访问 MCP 设置

1. 使用管理员账号登录 DooTask
2. 访问 AI 助手页面（通常在 `/ui/` 路径）
3. 在页面底部可以看到 "MCP 配置" 卡片

## 添加 MCP 配置

1. 点击 "MCP 配置" 卡片右上角的 "添加 MCP" 按钮
2. 在弹出的抽屉中填写以下信息：
   - **MCP 名称**: 为此 MCP 配置起一个易于识别的名称（例如：filesystem, database 等）
   - **MCP 配置**: 输入有效的 JSON 格式配置，例如：
     ```json
     {
       "command": "npx",
       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
     }
     ```
   - **支持的模型**: 按 AI 厂商分组显示所有可用的模型，勾选可以使用此 MCP 的具体模型（例如：gpt-4, claude-3-opus 等）
3. 点击 "保存" 按钮

## 编辑 MCP 配置

1. 在 MCP 配置列表中找到要编辑的配置
2. 点击该配置右侧的 "编辑" 按钮
3. 修改配置信息
4. 点击 "保存" 按钮

## 删除 MCP 配置

1. 在 MCP 配置列表中找到要删除的配置
2. 点击该配置右侧的 "删除" 按钮
3. 在确认对话框中点击 "确定"

## MCP 配置示例

### 文件系统访问

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/documents"]
}
```

### 数据库访问

```json
{
  "command": "node",
  "args": ["/path/to/database-mcp-server.js"]
}
```

### 自定义 MCP 服务器

```json
{
  "command": "python",
  "args": ["/path/to/custom-mcp-server.py", "--config", "config.json"]
}
```

## 配置文件存储

MCP 配置保存在项目根目录的 `mcp-config.json` 文件中。此文件已被添加到 `.gitignore`，不会被提交到版本控制系统。

## 注意事项

1. **权限要求**: 只有管理员可以配置 MCP
2. **JSON 格式**: 配置必须是有效的 JSON 格式，否则无法保存
3. **模型支持**: 为每个 MCP 配置选择合适的支持模型（具体的模型名称如 gpt-4, claude-3-opus 等），不同的 MCP 可能只适用于特定的 AI 模型
4. **模型分组显示**: 在编辑器中，模型按 AI 厂商（OpenAI, Claude, DeepSeek 等）分组显示，方便选择
5. **安全性**: 配置文件存储在本地，请确保服务器的安全性

## 常见问题

### Q: MCP 配置后何时生效？
A: MCP 配置保存后立即生效，无需重启服务。

### Q: 可以为同一个模型配置多个 MCP 吗？
A: 可以，每个 MCP 配置都是独立的，同一个具体模型（如 gpt-4）可以在多个 MCP 配置中勾选。

### Q: 配置文件在哪里？
A: 配置文件保存在项目根目录的 `mcp-config.json` 文件中。

### Q: 如何备份 MCP 配置？
A: 直接复制 `mcp-config.json` 文件即可。

## 技术实现

- **前端**: React + TypeScript
- **数据存储**: 本地 JSON 文件 (`mcp-config.json`)
- **API 端点**:
  - `GET /mcp/config` - 获取配置列表
  - `POST /mcp/config` - 保存配置列表

## 相关文件

- `/ui/src/components/aibot/MCPEditorSheet.tsx` - MCP 编辑器组件
- `/ui/src/components/aibot/MCPListCard.tsx` - MCP 列表展示组件
- `/ui/src/lib/mcp-storage.ts` - MCP 存储 API
- `/main.py` - 后端 API 路由（第 790-818 行）
