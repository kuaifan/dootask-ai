export interface SupportedModel {
  id: string
  name: string
}

export interface MCPConfig {
  id: string
  name: string
  config: string // JSON格式的MCP配置
  supportedModels: SupportedModel[] // 支持该MCP的模型列表
  enabled: boolean // 是否启用
  isSystem?: boolean // 是否为系统MCP（如DooTask）
}

export interface MCPConfigList {
  mcps?: MCPConfig[]
}

export const DOOTASK_MCP_ID = "dootask-mcp"

export const isSystemDooTaskMcp = (mcp?: Partial<MCPConfig>) => {
  if (!mcp) {
    return false
  }
  return Boolean(mcp.isSystem && mcp.id === DOOTASK_MCP_ID)
}

export const createMcpId = () => {
  const cryptoObj: Crypto | undefined = typeof globalThis !== "undefined" ? globalThis.crypto : undefined
  if (cryptoObj?.randomUUID) {
    return cryptoObj.randomUUID()
  }
  const randomSegment = () => Math.random().toString(36).slice(2, 10)
  return `mcp-${randomSegment()}${randomSegment()}`
}
