import { useMemo, useState, useCallback, useEffect } from "react"

import { Button } from "@/components/ui/button"
import { ModelListTable } from "@/components/aibot/ModelListTable"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"

import type { AIBotItem, AIBotKey } from "@/data/aibots"
import type { GeneratedField } from "@/lib/aibot"
import { parseModelNames } from "@/lib/aibot"
import { useI18n } from "@/lib/i18n-context"
export interface BotSettingsSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  bots: AIBotItem[]
  activeBot: AIBotKey
  onActiveBotChange: (value: AIBotKey) => void
  fieldMap: Record<AIBotKey, GeneratedField[]>
  formValues: Record<AIBotKey, Record<string, string>>
  initialValues: Record<AIBotKey, Record<string, string>>
  loadingMap: Record<AIBotKey, boolean>
  savingMap: Record<AIBotKey, boolean>
  defaultsLoadingMap: Record<AIBotKey, boolean>
  onReload: (bot: AIBotKey) => void
  onChangeField: (bot: AIBotKey, prop: string, value: string) => void
  onSubmit: (bot: AIBotKey) => void
  onReset: (bot: AIBotKey) => void
  onUseDefaultModels: (bot: AIBotKey) => Promise<string | null>
  onRegisterModelEditorBackHandler?: (handler: () => boolean) => void
}

export const BotSettingsSheet = ({
  activeBot,
  bots,
  fieldMap,
  formValues,
  initialValues,
  loadingMap,
  onActiveBotChange,
  onChangeField,
  onOpenChange,
  onReload,
  onReset,
  onSubmit,
  onUseDefaultModels,
  open,
  savingMap,
  defaultsLoadingMap,
  onRegisterModelEditorBackHandler,
}: BotSettingsSheetProps) => {
  const { t } = useI18n()
  const [modelEditor, setModelEditor] = useState<{
    bot: AIBotItem
    field: GeneratedField
  } | null>(null)
  const [modelEditorValue, setModelEditorValue] = useState("")

  const hasChanges = useMemo(() => {
    const result: Record<AIBotKey, boolean> = {} as Record<AIBotKey, boolean>
    bots.forEach((bot) => {
      const current = formValues[bot.value] ?? {}
      const initial = initialValues[bot.value] ?? {}
      const keys = new Set([...Object.keys(current), ...Object.keys(initial)])
      result[bot.value] = Array.from(keys).some(
        (key) => (current[key] ?? "") !== (initial[key] ?? ""),
      )
    })
    return result
  }, [bots, formValues, initialValues])

  const isModelEditorOpen = Boolean(modelEditor)
  const modelEditorOriginalValue = modelEditor
    ? formValues[modelEditor.bot.value]?.[modelEditor.field.prop] ?? ""
    : ""
  const modelEditorHasChanges = modelEditor
    ? modelEditorValue !== modelEditorOriginalValue
    : false
  const modelEditorSaving = modelEditor ? savingMap[modelEditor.bot.value] : false
  const modelEditorDefaultsLoading = modelEditor
    ? defaultsLoadingMap[modelEditor.bot.value]
    : false

  useEffect(() => {
    if (!open) {
      setModelEditor(null)
    }
  }, [open])

  useEffect(() => {
    if (!onRegisterModelEditorBackHandler) {
      return
    }
    const handler = () => {
      if (modelEditor) {
        setModelEditor(null)
        return true
      }
      return false
    }
    onRegisterModelEditorBackHandler(handler)
    return () => {
      onRegisterModelEditorBackHandler(() => false)
    }
  }, [modelEditor, onRegisterModelEditorBackHandler])

  const handleOpenModelEditor = useCallback(
    (bot: AIBotItem, field: GeneratedField) => {
      const currentValue = formValues[bot.value]?.[field.prop] ?? ""
      setModelEditor({ bot, field })
      setModelEditorValue(currentValue)
    },
    [formValues],
  )

  const handleCloseModelEditor = useCallback(() => {
    setModelEditor(null)
  }, [])

  const handleSaveModelEditor = useCallback(() => {
    if (!modelEditor || !modelEditorHasChanges) {
      setModelEditor(null)
      return
    }
    onChangeField(modelEditor.bot.value, modelEditor.field.prop, modelEditorValue)
    setModelEditor(null)
  }, [modelEditor, modelEditorHasChanges, modelEditorValue, onChangeField])

  const handleUseDefaultModelsInternal = useCallback(async () => {
    if (!modelEditor || modelEditorDefaultsLoading || modelEditorSaving) {
      return
    }
    const result = await onUseDefaultModels(modelEditor.bot.value)
    if (typeof result === "string") {
      setModelEditorValue(result)
    }
  }, [modelEditor, modelEditorDefaultsLoading, modelEditorSaving, onUseDefaultModels])

  const renderField = (bot: AIBotItem, field: GeneratedField) => {
    const fieldValue = formValues[bot.value]?.[field.prop] ?? ""
    const maxLength = field.maxlength
    const showWordLimit = field.showWordLimit
    const modelOptions = field.type === "model"
      ? parseModelNames(formValues[bot.value]?.[`${bot.value}_models`])
      : []

    if (field.originalProp === "models") {
      const displayModels = parseModelNames(fieldValue)
      return (
        <div key={field.prop} className="space-y-2">
          <Label className="text-sm font-medium">{field.label}</Label>
          <div className="space-y-3">
            <div className="rounded-md border bg-muted/30 p-3 text-sm max-h-44 overflow-y-auto">
              {displayModels.length ? (
                <ul className="space-y-2">
                  {displayModels.map((item) => (
                    <li key={`${item.value}|${item.label}`} className="leading-relaxed flex items-center justify-between">
                      <span className="font-medium">{item.label || item.value}</span>
                      {item.label && (
                        <span className="text-muted-foreground pl-2 text-xs text-right">{item.value}</span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">{t("sheet.models.empty")}</p>
              )}
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handleOpenModelEditor(bot, field)}
            >
              {t("sheet.models.edit")}
            </Button>
          </div>
          {(field.link || field.tip) && (
            <p className="text-xs text-muted-foreground break-all">
              {field.link ? (
                <>
                  {field.tipPrefix ?? t("sheet.tipPrefix")}{" "}
                  <a
                    href={field.link}
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary underline"
                  >
                    {field.link}
                  </a>
                </>
              ) : (
                field.tip
              )}
            </p>
          )}
        </div>
      )
    }

    const renderControl = () => {
      switch (field.type) {
        case "password":
          return (
            <Input
              type="password"
              value={fieldValue}
              onChange={(event) =>
                onChangeField(bot.value, field.prop, event.target.value)
              }
              placeholder={field.placeholder}
              maxLength={maxLength}
            />
          )
        case "textarea":
          return (
            <div className="space-y-2">
              <Textarea
                value={fieldValue}
                onChange={(event) =>
                  onChangeField(bot.value, field.prop, event.target.value)
                }
                placeholder={field.placeholder}
                maxLength={maxLength}
                rows={4}
              />
              {showWordLimit && maxLength && (
                <p className="text-xs text-muted-foreground">
                  {fieldValue.length}/{maxLength}
                </p>
              )}
              {field.functions && field.originalProp !== "models" && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={defaultsLoadingMap[bot.value]}
                  onClick={() => onUseDefaultModels(bot.value)}
                >
                  {defaultsLoadingMap[bot.value] ? t("sheet.fetching") : field.functions}
                </Button>
              )}
            </div>
          )
        case "model":
          return (
            <Select
              value={fieldValue}
              onValueChange={(value) => onChangeField(bot.value, field.prop, value)}
              disabled={modelOptions.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder={field.placeholder} />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )
        default:
          return (
            <Input
              value={fieldValue}
              onChange={(event) =>
                onChangeField(bot.value, field.prop, event.target.value)
              }
              placeholder={field.placeholder}
              maxLength={maxLength}
            />
          )
      }
    }

    return (
      <div key={field.prop} className="space-y-2">
        <Label className="text-sm font-medium">{field.label}</Label>
        {renderControl()}
        {(field.link || field.tip) && (
          <p className="text-xs text-muted-foreground break-all">
            {field.link ? (
              <>
                {field.tipPrefix ?? t("sheet.tipPrefix")}{" "}
                <a
                  href={field.link}
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary underline"
                >
                  {field.link}
                </a>
              </>
            ) : (
              field.tip
            )}
          </p>
        )}
      </div>
    )
  }

  return (
    <>
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent
          side="right"
          className="flex w-full max-w-2xl sm:max-w-4xl lg:max-w-4xl flex-col gap-6 overflow-hidden pt-[calc(var(--safe-area-top)+1.5rem)] pb-[calc(var(--safe-area-bottom)+1.5rem)]"
          onEscapeKeyDown={(event) => event.preventDefault()}
          onPointerDownOutside={(event) => event.preventDefault()}
        >
          <SheetHeader>
            <SheetTitle>{t("sheet.title")}</SheetTitle>
          </SheetHeader>
          <Tabs
            value={activeBot}
            onValueChange={(value) => onActiveBotChange(value as AIBotKey)}
            className="flex h-full flex-col overflow-hidden"
          >
            <ScrollArea className="w-full shrink-0 whitespace-nowrap pb-3">
              <TabsList className="inline-flex w-max gap-2">
                {bots.map((bot) => (
                  <TabsTrigger key={bot.value} value={bot.value} className="text-sm">
                    {bot.label}
                  </TabsTrigger>
                ))}
              </TabsList>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
            {bots.map((bot) => {
              const fields = fieldMap[bot.value] ?? []
              const isLoading = loadingMap[bot.value]
              return (
                <TabsContent
                  key={bot.value}
                  value={bot.value}
                  className="overflow-hidden data-[state=active]:min-h-0 data-[state=active]:flex"
                >
                  {isLoading ? (
                    <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                      {t("sheet.loading")}
                    </div>
                  ) : fields.length === 0 ? (
                    <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                      {t("sheet.empty")}
                    </div>
                  ) : (
                    <div className="flex flex-1 flex-col min-h-0">
                      <ScrollArea className="h-full">
                        <div className="flex flex-col gap-6 pb-10 pl-0.5 pr-3">
                          {fields.map((field) => renderField(bot, field))}
                        </div>
                      </ScrollArea>
                      <SheetFooter className="gap-3 border-t pt-4">
                        <div className="flex flex-1 flex-wrap justify-between gap-4">
                          <div className="flex items-center gap-3">
                            <Button
                              type="button"
                              variant="outline"
                              onClick={() => onReload(bot.value)}
                              disabled={loadingMap[bot.value] || savingMap[bot.value]}
                            >
                              {t("sheet.reload")}
                            </Button>
                            <Button
                              type="button"
                              variant="secondary"
                              onClick={() => onReset(bot.value)}
                              disabled={!hasChanges[bot.value]}
                            >
                              {t("sheet.reset")}
                            </Button>
                          </div>
                          <div className="flex items-center gap-3">
                            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
                              {t("sheet.back")}
                            </Button>
                            <Button
                              type="button"
                              onClick={() => onSubmit(bot.value)}
                              disabled={
                                savingMap[bot.value] ||
                                !hasChanges[bot.value]
                              }
                            >
                              {savingMap[bot.value] ? t("sheet.submitting") : t("sheet.submit")}
                            </Button>
                          </div>
                        </div>
                      </SheetFooter>
                    </div>
                  )}
                </TabsContent>
              )
            })}
          </Tabs>
        </SheetContent>
      </Sheet>
      <Sheet open={isModelEditorOpen} onOpenChange={(next) => !next && handleCloseModelEditor()}>
        <SheetContent
          side="right"
          className="flex w-full max-w-xl sm:max-w-3xl lg:max-w-3xl flex-col gap-0 overflow-hidden pt-[calc(var(--safe-area-top)+1.5rem)] pb-[calc(var(--safe-area-bottom)+1.5rem)]"
          onEscapeKeyDown={(event) => event.preventDefault()}
          onPointerDownOutside={(event) => event.preventDefault()}
        >
          <SheetHeader className="pb-6">
            <SheetTitle>{t("sheet.models.drawerTitle")}</SheetTitle>
            {modelEditor && (
              <SheetDescription>
                {t("sheet.models.drawerDescription")} {modelEditor.bot.label}
              </SheetDescription>
            )}
          </SheetHeader>
          {modelEditor && (
            <ScrollArea className="flex-1">
              <div className="space-y-4 pr-3 pb-6">
                <div className="space-y-2">
                  <Label className="text-sm font-medium">{modelEditor.field.label}</Label>
                  <ModelListTable
                    value={modelEditorValue}
                    onChange={setModelEditorValue}
                    modelLabel={t("sheet.models.column.model")}
                    displayLabel={t("sheet.models.column.label")}
                    actionLabel={t("sheet.models.column.actions")}
                    addButtonLabel={t("sheet.models.add")}
                    emptyLabel={t("sheet.models.empty")}
                    removeLabel={t("sheet.models.remove")}
                    modelPlaceholder={modelEditor.field.placeholder ?? t("sheet.models.modelPlaceholder")}
                    labelPlaceholder={t("sheet.models.labelPlaceholder")}
                    maxLength={modelEditor.field.maxlength}
                    disabled={modelEditorDefaultsLoading || modelEditorSaving}
                  />
                </div>
              </div>
              <ScrollBar orientation="vertical" />
            </ScrollArea>
          )}
          <SheetFooter className="gap-3 border-t pt-4">
            <div className="flex flex-1 flex-wrap justify-between gap-4">
              <div className="flex items-center gap-3">
                {modelEditor?.field.functions && (
                  <Button
                    type="button"
                    variant="outline"
                    disabled={modelEditorDefaultsLoading || modelEditorSaving}
                    onClick={handleUseDefaultModelsInternal}
                  >
                    {modelEditorDefaultsLoading
                      ? t("sheet.fetching")
                      : modelEditor.field.functions}
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-3">
                <Button type="button" variant="outline" onClick={handleCloseModelEditor}>
                  {t("sheet.models.cancel")}
                </Button>
                <Button
                  type="button"
                  onClick={handleSaveModelEditor}
                  disabled={!modelEditorHasChanges || modelEditorSaving}
                >
                  {t("sheet.models.save")}
                </Button>
              </div>
            </div>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  )
}
