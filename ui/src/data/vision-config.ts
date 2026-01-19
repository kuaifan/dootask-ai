// ui/src/data/vision-config.ts
export interface SupportedModel {
  id: string
  name: string
}

export interface VisionConfig {
  enabled: boolean
  supportedModels: SupportedModel[]
  maxImageSize: number
  maxFileSize: number
  compressionQuality: number
  availableModels?: SupportedModel[] // Computed, returned by API
}

export const DEFAULT_VISION_CONFIG: VisionConfig = {
  enabled: false,
  supportedModels: [],
  maxImageSize: 2048,
  maxFileSize: 10,
  compressionQuality: 80,
}
