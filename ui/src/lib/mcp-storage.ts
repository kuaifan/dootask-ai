import type { MCPConfig, MCPConfigList } from "@/data/mcp-config"

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
    return data.mcps || []
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
  const index = existingMcps.findIndex((m) => m.name === mcp.name)
  let newMcps: MCPConfig[]

  if (index >= 0) {
    // 更新现有配置
    newMcps = [...existingMcps]
    newMcps[index] = mcp
  } else {
    // 添加新配置
    newMcps = [...existingMcps, mcp]
  }

  await saveMCPConfigs(newMcps)
  return newMcps
}

/**
 * 删除MCP配置
 */
export const deleteMCPConfig = async (name: string, existingMcps: MCPConfig[]): Promise<MCPConfig[]> => {
  const newMcps = existingMcps.filter((m) => m.name !== name)
  await saveMCPConfigs(newMcps)
  return newMcps
}
