import type { AIBotItem } from "@/data/aibots"
import type { FieldConfig, BotConfig } from "@/data/aibot-config"

export interface GeneratedField extends FieldConfig {
  prop: string
  originalProp: string
}

export interface ModelItem {
  id: string
  name: string
  support_mcp: boolean
}

export const parseModelNames = (raw: string | ModelItem[] | undefined | null) => {
  if (!raw) return []

  // 如果已经是数组格式（新的 JSON 格式）
  if (Array.isArray(raw)) {
    return raw.map(item => ({
      value: item.id,
      label: item.name,
      support_mcp: item.support_mcp
    }))
  }

  // 兼容旧的字符串格式 "id | name"
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [value, label] = line.split("|").map((item) => item.trim())
      return {
        value,
        label: label || value,
        support_mcp: false
      }
    })
    .filter((item) => item.value)
}

export const mergeFields = (
  baseFields: FieldConfig[],
  botConfig: BotConfig | undefined,
  type: AIBotItem["value"],
): GeneratedField[] => {
  const prefixed = baseFields.map((field) => ({
    ...field,
    prop: `${type}_${field.prop}`,
    originalProp: field.prop,
  }))

  botConfig?.extraFields?.forEach((extra) => {
    const targetProp = `${type}_${extra.prop}`
    const existingIndex = prefixed.findIndex((field) => field.prop === targetProp)
    const newField = {
      ...extra,
      prop: targetProp,
      originalProp: extra.prop,
    } as Partial<GeneratedField>

    if (existingIndex >= 0) {
      const definedEntries = Object.entries(newField).filter(
        ([, value]) => value !== undefined,
      )
      prefixed[existingIndex] = Object.assign(
        {},
        prefixed[existingIndex],
        Object.fromEntries(definedEntries),
      )
    } else if (extra.after) {
      const afterIndex = prefixed.findIndex(
        (field) => field.prop === `${type}_${extra.after}`,
      )
      if (afterIndex >= 0) {
        prefixed.splice(afterIndex + 1, 0, {
          label: extra.label ?? extra.prop,
          ...newField,
        } as GeneratedField)
        return
      }
      prefixed.push({
        label: extra.label ?? extra.prop,
        ...newField,
      } as GeneratedField)
    } else {
      prefixed.push({
        label: extra.label ?? extra.prop,
        ...newField,
      } as GeneratedField)
    }
  })

  let sortIndex = 999999
  prefixed.forEach((field) => {
    if (typeof field.sort === "undefined") {
      field.sort = ++sortIndex
    }
  })

  return prefixed.sort((a, b) => (a.sort ?? 0) - (b.sort ?? 0))
}
