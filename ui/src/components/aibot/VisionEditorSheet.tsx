// ui/src/components/aibot/VisionEditorSheet.tsx
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Switch } from "@/components/ui/switch"
import type { VisionConfig } from "@/data/vision-config"
import type { AIBotItem } from "@/data/aibots"
import { useEffect, useState } from "react"

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

  const toggleModel = (model: { value: string; label: string; support_vision: boolean }) => {
    if (!model.support_vision) return

    const isSelected = isModelSelected(model.value)
    if (isSelected) {
      setEditConfig({
        ...editConfig,
        supportedModels: editConfig.supportedModels.filter((m) => m.id !== model.value),
      })
    } else {
      setEditConfig({
        ...editConfig,
        supportedModels: [
          ...editConfig.supportedModels,
          { id: model.value, name: model.label },
        ],
      })
    }
  }

  // Group models by provider, only show those with support_vision capability
  const modelsByProvider = aiBots
    .map((bot) => ({
      provider: bot.value,
      label: bot.label,
      models: (bot.models || []).filter((m) => m.support_vision),
    }))
    .filter((group) => group.models.length > 0)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{t("vision.editTitle")}</SheetTitle>
          <SheetDescription>{t("vision.description")}</SheetDescription>
        </SheetHeader>

        <div className="space-y-6 py-4">
          {/* Enable Switch */}
          <div className="flex items-center justify-between">
            <Label htmlFor="vision-enabled">{t("vision.enabled")}</Label>
            <Switch
              id="vision-enabled"
              checked={editConfig.enabled}
              onCheckedChange={(checked) =>
                setEditConfig({ ...editConfig, enabled: checked })
              }
            />
          </div>

          {/* Supported Models */}
          <div className="space-y-3">
            <Label>{t("vision.supportedModels")}</Label>
            <p className="text-sm text-muted-foreground">{t("vision.supportedModelsTip")}</p>
            <div className="space-y-4 max-h-64 overflow-y-auto border rounded-md p-3">
              {modelsByProvider.map((group) => (
                <div key={group.provider} className="space-y-2">
                  <div className="text-sm font-medium text-muted-foreground">
                    {group.label}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {group.models.map((model) => (
                      <div key={model.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`model-${model.value}`}
                          checked={isModelSelected(model.value)}
                          onCheckedChange={() => toggleModel(model)}
                        />
                        <label
                          htmlFor={`model-${model.value}`}
                          className="text-sm cursor-pointer"
                        >
                          {model.label}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Max Image Size */}
          <div className="space-y-2">
            <Label htmlFor="max-image-size">
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
            <Label htmlFor="max-file-size">
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
            <Label htmlFor="compression-quality">
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

        <SheetFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("vision.cancel")}
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {t("vision.save")}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
