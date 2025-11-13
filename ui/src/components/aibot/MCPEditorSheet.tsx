import { useState, useEffect } from "react"

import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Switch } from "@/components/ui/switch"
import { ScrollArea } from "@/components/ui/scroll-area"

import type { MCPConfig } from "@/data/mcp-config"
import type { AIBotItem } from "@/data/aibots"
import { useI18n } from "@/lib/i18n-context"

export interface MCPEditorSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  mcp: MCPConfig | null
  bots: AIBotItem[]
  allModels: Record<string, string> // key: modelValue, value: modelLabel
  onSave: (mcp: MCPConfig) => void
}

export const MCPEditorSheet = ({
  open,
  onOpenChange,
  mcp,
  bots,
  allModels,
  onSave,
}: MCPEditorSheetProps) => {
  const { t } = useI18n()
  const [name, setName] = useState("")
  const [config, setConfig] = useState("")
  const [supportedModels, setSupportedModels] = useState<string[]>([])
  const [enabled, setEnabled] = useState(true)
  const [configError, setConfigError] = useState("")

  // 将所有模型按AI厂商分组
  const modelsByBot = bots.reduce((acc, bot) => {
    const models = bot.tags || []
    if (models.length > 0) {
      acc[bot.value] = {
        label: bot.label,
        models: models.map(tag => {
          // 从 allModels 中找到对应的模型值
          const modelEntry = Object.entries(allModels).find(([_, label]) => label === tag)
          return {
            value: modelEntry?.[0] || tag,
            label: tag
          }
        })
      }
    }
    return acc
  }, {} as Record<string, { label: string, models: { value: string, label: string }[] }>)

  useEffect(() => {
    if (mcp) {
      setName(mcp.name)
      setConfig(mcp.config)
      setSupportedModels(mcp.supportedModels)
      setEnabled(mcp.enabled)
    } else {
      setName("")
      setConfig("")
      setSupportedModels([])
      setEnabled(true)
    }
    setConfigError("")
  }, [mcp, open])

  const validateConfig = (configText: string): boolean => {
    if (!configText.trim()) {
      setConfigError("")
      return true
    }
    try {
      JSON.parse(configText)
      setConfigError("")
      return true
    } catch (error) {
      setConfigError(t("mcp.configInvalid"))
      return false
    }
  }

  const handleConfigChange = (value: string) => {
    setConfig(value)
    validateConfig(value)
  }

  const handleToggleModel = (modelValue: string, checked: boolean) => {
    if (checked) {
      setSupportedModels((prev) => [...prev, modelValue])
    } else {
      setSupportedModels((prev) => prev.filter((m) => m !== modelValue))
    }
  }

  const handleSave = () => {
    if (!name.trim()) {
      return
    }
    if (!validateConfig(config)) {
      return
    }
    onSave({
      name: name.trim(),
      config,
      supportedModels,
      enabled,
      isSystem: mcp?.isSystem,
    })
    onOpenChange(false)
  }

  const hasChanges = mcp
    ? name !== mcp.name ||
      config !== mcp.config ||
      enabled !== mcp.enabled ||
      JSON.stringify(supportedModels.sort()) !== JSON.stringify(mcp.supportedModels.sort())
    : name.trim() !== "" || config.trim() !== "" || supportedModels.length > 0 || !enabled

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex w-full max-w-2xl sm:max-w-4xl lg:max-w-4xl flex-col gap-6 overflow-hidden"
        onEscapeKeyDown={(event) => event.preventDefault()}
      >
        <SheetHeader>
          <SheetTitle>{mcp ? t("mcp.editTitle") : t("mcp.addTitle")}</SheetTitle>
          <SheetDescription>{t("mcp.description")}</SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-6 pl-0.5 pr-3 pb-6">
            {/* MCP名称 */}
            <div className="space-y-2">
              <Label htmlFor="mcp-name" className="text-sm font-medium">
                {t("mcp.name")}
              </Label>
              <Input
                id="mcp-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("mcp.namePlaceholder")}
                maxLength={100}
                disabled={mcp?.isSystem}
              />
            </div>

            {/* MCP配置 - 系统MCP不显示 */}
            {!mcp?.isSystem && (
              <div className="space-y-2">
                <Label htmlFor="mcp-config" className="text-sm font-medium">
                  {t("mcp.config")}
                </Label>
                <Textarea
                  id="mcp-config"
                  value={config}
                  onChange={(e) => handleConfigChange(e.target.value)}
                  placeholder={t("mcp.configPlaceholder")}
                  rows={10}
                  className={configError ? "border-red-500" : ""}
                />
                {configError && (
                  <p className="text-xs text-red-500">{configError}</p>
                )}
                <p className="text-xs text-muted-foreground">
                  {t("mcp.configTip")}
                </p>
              </div>
            )}

            {/* 支持的模型 */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">
                {t("mcp.supportedModels")}
              </Label>
              <div className="space-y-4 rounded-md border p-4 max-h-96 overflow-y-auto">
                {Object.entries(modelsByBot).map(([botValue, botInfo]) => (
                  <div key={botValue} className="space-y-2">
                    <div className="font-medium text-sm text-muted-foreground">
                      {botInfo.label}
                    </div>
                    <div className="space-y-2 pl-4">
                      {botInfo.models.map((model) => (
                        <Checkbox
                          key={model.value}
                          id={`model-${model.value}`}
                          checked={supportedModels.includes(model.value)}
                          onCheckedChange={(checked) => handleToggleModel(model.value, checked)}
                        >
                          {model.label}
                        </Checkbox>
                      ))}
                    </div>
                  </div>
                ))}
                {Object.keys(modelsByBot).length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    {t("sheet.models.empty")}
                  </p>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                {t("mcp.supportedModelsTip")}
              </p>
            </div>
          </div>
        </ScrollArea>

        <SheetFooter className="gap-3 border-t pt-4">
          <div className="flex flex-1 flex-col gap-2 sm:flex-row sm:justify-between sm:gap-4">
            <div className="flex items-center gap-3">
              <Label htmlFor="mcp-enabled" className="text-sm font-medium">
                {t("mcp.enabled")}
              </Label>
              <Switch
                id="mcp-enabled"
                checked={enabled}
                onCheckedChange={setEnabled}
                disabled={mcp?.isSystem}
              />
            </div>
            <div className="flex items-center gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                {t("mcp.cancel")}
              </Button>
              <Button
                type="button"
                onClick={handleSave}
                disabled={!name.trim() || !!configError || !hasChanges}
              >
                {t("mcp.save")}
              </Button>
            </div>
          </div>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
