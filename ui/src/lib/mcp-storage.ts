import {
  type MCPConfig,
  type MCPConfigList,
  createMcpId,
  DOOTASK_MCP_ID,
  isSystemDooTaskMcp,
} from "@/data/mcp-config"

const normalizeMcpConfig = (mcp: Partial<MCPConfig>): MCPConfig => {
  const normalizedId = isSystemDooTaskMcp(mcp) ? DOOTASK_MCP_ID : mcp.id ?? createMcpId()
  return {
    id: normalizedId,
    name: typeof mcp.name === "string" ? mcp.name : "",
    config: typeof mcp.config === "string" ? mcp.config : "",
    supportedModels: Array.isArray(mcp.supportedModels) ? mcp.supportedModels : [],
    enabled: mcp.enabled ?? true,
    isSystem: mcp.isSystem,
  }
}

const ensureMcpIds = (mcps: Partial<MCPConfig>[] = []) => mcps.map((mcp) => normalizeMcpConfig(mcp))

/**
 * 从本地文件加载MCP配置列表
 */
export const loadMCPConfigs = async (): Promise<MCPConfig[]> => {
  try {
    const response = await fetch(`/ai/mcp/config`)
    if (!response.ok) {
      if (response.status === 404) {
        return []
      }
      throw new Error(`Failed to load MCP configs: ${response.statusText}`)
    }
    const data: MCPConfigList = await response.json()
    return ensureMcpIds(data.mcps)
  } catch (error) {
    console.error("Error loading MCP configs:", error)
    return []
  }
}

/**
 * 保存MCP配置列表到本地文件
 */
export const saveMCPConfigs = async (mcps: MCPConfig[]): Promise<void> => {
  try {
    const response = await fetch(`/ai/mcp/config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mcps }),
    })
    if (!response.ok) {
      throw new Error(`Failed to save MCP configs: ${response.statusText}`)
    }
  } catch (error) {
    console.error("Error saving MCP configs:", error)
    throw error
  }
}

/**
 * 添加或更新MCP配置
 */
export const saveMCPConfig = async (mcp: MCPConfig, existingMcps: MCPConfig[]): Promise<MCPConfig[]> => {
  const normalizedTarget = normalizeMcpConfig(mcp)
  const index = existingMcps.findIndex((existing) => existing.id === normalizedTarget.id)
  const isSystemTarget = isSystemDooTaskMcp(normalizedTarget)
  const target = isSystemTarget
    ? {
        ...(index >= 0 ? existingMcps[index] : normalizedTarget),
        supportedModels: normalizedTarget.supportedModels,
      }
    : normalizedTarget
  let newMcps: MCPConfig[]

  if (index >= 0) {
    // 更新现有配置
    newMcps = [...existingMcps]
    newMcps[index] = target
  } else {
    // 添加新配置
    newMcps = [...existingMcps, target]
  }

  await saveMCPConfigs(newMcps)
  return newMcps
}

/**
 * 删除MCP配置
 */
export const deleteMCPConfig = async (id: string, existingMcps: MCPConfig[]): Promise<MCPConfig[]> => {
  const newMcps = existingMcps.filter((m) => m.id !== id)
  await saveMCPConfigs(newMcps)
  return newMcps
}
