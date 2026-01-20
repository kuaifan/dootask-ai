// ui/src/lib/vision-storage.ts
import { type VisionConfig, DEFAULT_VISION_CONFIG } from "@/data/vision-config"

/**
 * Load vision configuration from API
 */
export const loadVisionConfig = async (): Promise<VisionConfig> => {
  try {
    const response = await fetch(`/ai/vision/config`)
    if (!response.ok) {
      if (response.status === 404) {
        return DEFAULT_VISION_CONFIG
      }
      throw new Error(`Failed to load vision config: ${response.statusText}`)
    }
    const result = await response.json()
    if (result.code === 200 && result.data) {
      return {
        ...DEFAULT_VISION_CONFIG,
        ...result.data,
      }
    }
    return DEFAULT_VISION_CONFIG
  } catch (error) {
    console.error("Error loading vision config:", error)
    return DEFAULT_VISION_CONFIG
  }
}

/**
 * Save vision configuration to API
 */
export const saveVisionConfig = async (config: VisionConfig): Promise<boolean> => {
  try {
    const response = await fetch(`/ai/vision/config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    })
    if (!response.ok) {
      throw new Error(`Failed to save vision config: ${response.statusText}`)
    }
    const result = await response.json()
    return result.code === 200
  } catch (error) {
    console.error("Error saving vision config:", error)
    throw error
  }
}
