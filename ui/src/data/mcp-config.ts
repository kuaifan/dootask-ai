export interface MCPConfig {
  name: string
  config: string // JSON格式的MCP配置
  supportedModels: string[] // 支持该MCP的模型列表
  enabled: boolean // 是否启用
  isSystem?: boolean // 是否为系统MCP（如DooTask）
}

export interface MCPConfigList {
  mcps: MCPConfig[]
}
