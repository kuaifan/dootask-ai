// ui/src/components/aibot/VisionEditorSheet.tsx
import { useState, useEffect, useMemo } from "react"

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
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Switch } from "@/components/ui/switch"
import { ScrollArea } from "@/components/ui/scroll-area"

import type { VisionConfig } from "@/data/vision-config"
import type { AIBotItem } from "@/data/aibots"

interface VisionEditorSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  config: VisionConfig
  onSave: (config: VisionConfig) => Promise<void>
  aiBots: AIBotItem[]
  t: (key: string) => string
}

export function VisionEditorSheet({
  open,
  onOpenChange,
  config,
  onSave,
  aiBots,
  t,
}: VisionEditorSheetProps) {
  const [editConfig, setEditConfig] = useState<VisionConfig>(config)
  const [saving, setSaving] = useState(false)

  // Group all models by provider (like MCPEditorSheet)
  const modelsByBot = useMemo(() => {
    return aiBots.reduce((acc, bot) => {
      const models = bot.models ?? []
      if (models.length > 0) {
        acc[bot.value] = {
          label: bot.label,
          models: models.map((model) => ({
            value: model.value,
            label: model.label,
          })),
        }
      }
      return acc
    }, {} as Record<string, { label: string; models: { value: string; label: string }[] }>)
  }, [aiBots])

  useEffect(() => {
    if (open) {
      setEditConfig(config)
    }
  }, [open, config])

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(editConfig)
      onOpenChange(false)
    } finally {
      setSaving(false)
    }
  }

  const isModelSelected = (modelId: string) => {
    return editConfig.supportedModels.some((m) => m.id === modelId)
  }

  const handleToggleModel = (model: { value: string; label: string }, checked: boolean | string) => {
    const isChecked = checked === true
    if (isChecked) {
      setEditConfig((prev) => {
        if (prev.supportedModels.some((m) => m.id === model.value)) {
          return prev
        }
        return {
          ...prev,
          supportedModels: [...prev.supportedModels, { id: model.value, name: model.label }],
        }
      })
      return
    }
    setEditConfig((prev) => ({
      ...prev,
      supportedModels: prev.supportedModels.filter((m) => m.id !== model.value),
    }))
  }

  const normalizeModels = (models: { id: string; name: string }[]) =>
    [...models]
      .map((model) => ({ id: model.id, name: model.name }))
      .sort((a, b) => a.id.localeCompare(b.id))

  const hasChanges =
    editConfig.enabled !== config.enabled ||
    editConfig.maxImageSize !== config.maxImageSize ||
    editConfig.maxFileSize !== config.maxFileSize ||
    editConfig.compressionQuality !== config.compressionQuality ||
    JSON.stringify(normalizeModels(editConfig.supportedModels)) !==
      JSON.stringify(normalizeModels(config.supportedModels))

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex w-full max-w-2xl sm:max-w-4xl lg:max-w-4xl flex-col gap-6 overflow-hidden pt-[calc(var(--safe-area-top)+1.5rem)] pb-[calc(var(--safe-area-bottom)+1.5rem)]"
        onEscapeKeyDown={(event) => event.preventDefault()}
        onPointerDownOutside={(event) => event.preventDefault()}
      >
        <SheetHeader>
          <SheetTitle>{t("vision.editTitle")}</SheetTitle>
          <SheetDescription>{t("vision.description")}</SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-6 pl-0.5 pr-3 pb-6">
            {/* Supported Models */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">
                {t("vision.supportedModels")}
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
                          checked={isModelSelected(model.value)}
                          onCheckedChange={(checked) => handleToggleModel(model, checked)}
                        >
                          {model.label}
                        </Checkbox>
                      ))}
                    </div>
                  </div>
                ))}
                {Object.keys(modelsByBot).length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    {t("vision.noModelsAvailable")}
                  </p>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                {t("vision.supportedModelsTip")}
              </p>
            </div>

            {/* Image Limits Section */}
            <div className="space-y-4">
              <Label className="text-sm font-medium">
                {t("vision.imageLimit")}
              </Label>

              {/* Max Image Size */}
              <div className="space-y-2">
                <Label htmlFor="max-image-size" className="text-sm text-muted-foreground">
                  {t("vision.maxImageSize")} ({t("vision.maxImageSizeTip")})
                </Label>
                <Input
                  id="max-image-size"
                  type="number"
                  min={256}
                  max={8192}
                  value={editConfig.maxImageSize}
                  onChange={(e) =>
                    setEditConfig({
                      ...editConfig,
                      maxImageSize: parseInt(e.target.value) || 2048,
                    })
                  }
                />
              </div>

              {/* Max File Size */}
              <div className="space-y-2">
                <Label htmlFor="max-file-size" className="text-sm text-muted-foreground">
                  {t("vision.maxFileSize")} ({t("vision.maxFileSizeTip")})
                </Label>
                <Input
                  id="max-file-size"
                  type="number"
                  min={1}
                  max={50}
                  value={editConfig.maxFileSize}
                  onChange={(e) =>
                    setEditConfig({
                      ...editConfig,
                      maxFileSize: parseInt(e.target.value) || 10,
                    })
                  }
                />
              </div>

              {/* Compression Quality */}
              <div className="space-y-2">
                <Label htmlFor="compression-quality" className="text-sm text-muted-foreground">
                  {t("vision.compressionQuality")} ({t("vision.compressionQualityTip")})
                </Label>
                <Input
                  id="compression-quality"
                  type="number"
                  min={1}
                  max={100}
                  value={editConfig.compressionQuality}
                  onChange={(e) =>
                    setEditConfig({
                      ...editConfig,
                      compressionQuality: parseInt(e.target.value) || 80,
                    })
                  }
                />
              </div>
            </div>
          </div>
        </ScrollArea>

        <SheetFooter className="gap-3 border-t pt-4">
          <div className="flex flex-1 flex-wrap justify-between gap-4">
            <div className="flex items-center gap-3">
              <Label htmlFor="vision-enabled" className="text-sm font-medium">
                {t("vision.enabled")}
              </Label>
              <Switch
                id="vision-enabled"
                checked={editConfig.enabled}
                onCheckedChange={(checked) =>
                  setEditConfig({ ...editConfig, enabled: checked })
                }
              />
            </div>
            <div className="flex items-center gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                {t("vision.cancel")}
              </Button>
              <Button
                type="button"
                onClick={handleSave}
                disabled={saving || !hasChanges}
              >
                {t("vision.save")}
              </Button>
            </div>
          </div>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
