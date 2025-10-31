import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import {
  appReady,
  getUserInfo,
  modalError,
  modalInfo,
  messageError,
  messageSuccess,
  openDialogUserid,
  requestAPI,
  interceptBack,
} from "@dootask/tools"

import { BotCard } from "@/components/aibot/BotCard"
import { BotSettingsSheet } from "@/components/aibot/BotSettingsSheet"
import type { AIBotItem, AIBotKey } from "@/data/aibots"
import { createLocalizedAIBotList } from "@/data/aibots"
import { getAISystemConfig, type SystemConfig } from "@/data/aibot-config"
import { mergeFields, parseModelNames } from "@/lib/aibot"
import type { GeneratedField } from "@/lib/aibot"
import { useI18n } from "@/lib/i18n-context"

type SettingsState = Record<AIBotKey, Record<string, string>>
type LoadingState = Record<AIBotKey, boolean>

const getThemeFromSearch = () => {
  const params = new URLSearchParams(window.location.search)
  return params.get("theme") === "dark" ? "dark" : "light"
}

const applyTheme = (theme: "dark" | "light") => {
  const root = document.documentElement
  if (theme === "dark") {
    root.classList.add("dark")
    root.setAttribute("data-theme", "dark")
  } else {
    root.classList.remove("dark")
    root.setAttribute("data-theme", "light")
  }
}

const fieldMapFactory = (
  bots: AIBotItem[],
  config: SystemConfig,
): Record<AIBotKey, GeneratedField[]> => {
  const baseFields = config.fields
  return bots.reduce((acc, bot) => {
    acc[bot.value] = mergeFields(baseFields, config.aiList[bot.value], bot.value)
    return acc
  }, {} as Record<AIBotKey, GeneratedField[]>)
}

const emptyState = {} as SettingsState

const resolveErrorMessage = (error: unknown, fallback: string) => {
  if (error && typeof error === "object") {
    if ("msg" in error && error.msg) {
      return String(error.msg)
    }
    if ("message" in error && error.message) {
      return String(error.message)
    }
  }
  if (error instanceof Error && error.message) {
    return error.message
  }
  return fallback
}

function App() {
  const { lang, t } = useI18n()
  const systemConfig = useMemo(() => getAISystemConfig(lang), [lang])
  const [bots, setBots] = useState<AIBotItem[]>(() => createLocalizedAIBotList(lang))
  const [chatLoading, setChatLoading] = useState<LoadingState>({} as LoadingState)
  const [isAdmin, setIsAdmin] = useState(false)
  const [settingsOpen, setSettingsOpenState] = useState(false)
  const [activeBot, setActiveBot] = useState<AIBotKey>("openai")
  const [formValues, setFormValues] = useState<SettingsState>(emptyState)
  const [initialValues, setInitialValues] = useState<SettingsState>(emptyState)
  const [settingsLoadingMap, setSettingsLoadingMap] = useState<LoadingState>({} as LoadingState)
  const [settingsSavingMap, setSettingsSavingMap] = useState<LoadingState>({} as LoadingState)
  const [defaultsLoading, setDefaultsLoading] = useState<LoadingState>({} as LoadingState)

  const settingsOpenRef = useRef(settingsOpen)
  const interceptReleaseRef = useRef<(() => void) | null>(null)
  const modelEditorBackHandlerRef = useRef<() => boolean>(() => false)

  const fieldMap = useMemo(() => fieldMapFactory(bots, systemConfig), [bots, systemConfig])

  useEffect(() => {
    settingsOpenRef.current = settingsOpen
  }, [settingsOpen])

  useEffect(() => {
    setBots((prev) => createLocalizedAIBotList(lang, prev))
  }, [lang])

  useEffect(() => {
    applyTheme(getThemeFromSearch())
  }, [])

  useEffect(() => {
    const init = async () => {
      try {
        await appReady()
      } catch {
        // ignore; best effort
      }

      try {
        const user = await getUserInfo()
        if (user?.identity?.includes("admin")) {
          setIsAdmin(true)
        }
      } catch {
        // cannot determine admin state, keep default false
      }

      await refreshBotTags()
    }

    init().catch((error) => {
      console.error("Failed to initialize AI bot UI", error)
    })
  }, [])

  const refreshBotTags = async () => {
    try {
      const { data } = await requestAPI({
        url: "system/setting/aibot_models",
        method: "get",
      })
      if (!data || typeof data !== "object") {
        return
      }

      setBots((prev) =>
        prev.map((bot) => {
          const modelsRaw = data?.[`${bot.value}_models`]
          const defaultModel = data?.[`${bot.value}_model`]
          const options = parseModelNames(modelsRaw)
          const tagLabel =
            (options.find((option) => option.value === defaultModel)?.label ?? defaultModel) ||
            options[0]?.label

          return {
            ...bot,
            tags: options.map((option) => option.label),
            tagLabel: tagLabel ?? undefined,
          }
        }),
      )
    } catch (error) {
      console.error("Failed to fetch AI bot models", error)
    }
  }

  const handleShowDescription = (bot: AIBotItem) => {
    modalInfo(bot.desc)
  }

  const handleStartChat = async (bot: AIBotItem) => {
    setChatLoading((prev) => ({ ...prev, [bot.value]: true }))
    try {
      const { data } = await requestAPI({
        url: "users/search/ai",
        method: "get",
        data: { type: bot.value },
      })
      if (!data?.userid) {
        throw new Error(t("errors.botNotFound"))
      }
      await openDialogUserid(Number(data.userid))
    } catch (error) {
      messageError(resolveErrorMessage(error, t("errors.botUnavailable")))
    } finally {
      setChatLoading((prev) => ({ ...prev, [bot.value]: false }))
    }
  }

  const loadSettings = async (bot: AIBotKey, force = false) => {
    if (!force && formValues[bot]) {
      return
    }
    setSettingsLoadingMap((prev) => ({ ...prev, [bot]: true }))
    try {
      const { data } = await requestAPI({
        url: "system/setting/aibot",
        method: "get",
        data: {
          type: "get",
          filter: bot,
        },
      })
      const payload = (data ?? {}) as Record<string, string>
      setFormValues((prev) => ({ ...prev, [bot]: payload }))
      setInitialValues((prev) => ({ ...prev, [bot]: payload }))
    } catch (error) {
      messageError(resolveErrorMessage(error, t("errors.loadFailed")))
    } finally {
      setSettingsLoadingMap((prev) => ({ ...prev, [bot]: false }))
    }
  }

  const ensureIntercept = useCallback(async () => {
    if (interceptReleaseRef.current) {
      return
    }
    try {
      interceptReleaseRef.current = await interceptBack(() => {
        if (modelEditorBackHandlerRef.current && modelEditorBackHandlerRef.current()) {
          return true
        }
        if (settingsOpenRef.current) {
          setSettingsOpenState(false)
          return true
        }
        return false
      })
    } catch (error) {
      console.error("Failed to register interceptBack", error)
    }
  }, [])

  const releaseIntercept = useCallback(() => {
    if (interceptReleaseRef.current) {
      try {
        interceptReleaseRef.current()
      } catch (error) {
        console.error("Failed to release interceptBack", error)
      }
      interceptReleaseRef.current = null
    }
    modelEditorBackHandlerRef.current = () => false
  }, [])

  const handleRegisterModelEditorBackHandler = useCallback((handler: () => boolean) => {
    modelEditorBackHandlerRef.current = handler
  }, [])

  useEffect(() => {
    if (isAdmin && settingsOpen) {
      void ensureIntercept()
    } else if (!settingsOpen) {
      releaseIntercept()
    }
  }, [ensureIntercept, isAdmin, releaseIntercept, settingsOpen])

  useEffect(() => {
    return () => {
      releaseIntercept()
    }
  }, [releaseIntercept])

  const handleOpenSettings = async (bot: AIBotItem) => {
    if (!isAdmin) {
      messageError(t("errors.adminOnly"))
      return
    }
    setActiveBot(bot.value)
    setSettingsOpenState(true)
    await loadSettings(bot.value)
  }

  const handleTabChange = async (value: AIBotKey) => {
    setActiveBot(value)
    await loadSettings(value)
  }

  const handleChangeField = (bot: AIBotKey, prop: string, value: string) => {
    setFormValues((prev) => ({
      ...prev,
      [bot]: {
        ...(prev[bot] ?? {}),
        [prop]: value,
      },
    }))
  }

  const handleReset = (bot: AIBotKey) => {
    const original = initialValues[bot] ?? {}
    setFormValues((prev) => ({
      ...prev,
      [bot]: { ...original },
    }))
  }

  const handleReload = async (bot: AIBotKey) => {
    await loadSettings(bot, true)
  }

  const handleSubmit = async (bot: AIBotKey) => {
    const fields = fieldMap[bot] ?? []
    if (!fields.length) {
      messageError(t("errors.botUnsupported"))
      return
    }
    const payload = fields.reduce<Record<string, string>>((acc, field) => {
      acc[field.prop] = formValues[bot]?.[field.prop] ?? ""
      return acc
    }, {})

    setSettingsSavingMap((prev) => ({ ...prev, [bot]: true }))
    try {
      const response = await requestAPI({
        url: "system/setting/aibot",
        method: "post",
        data: {
          ...payload,
          type: "save",
          filter: bot,
        },
      })
      const savedData = (response.data ?? {}) as Record<string, string>
      setFormValues((prev) => ({ ...prev, [bot]: savedData }))
      setInitialValues((prev) => ({ ...prev, [bot]: savedData }))
      messageSuccess(response.msg ?? t("success.save"))
      await refreshBotTags()
    } catch (error) {
      modalError(resolveErrorMessage(error, t("errors.submitFailed")))
    } finally {
      setSettingsSavingMap((prev) => ({ ...prev, [bot]: false }))
    }
  }

  const handleUseDefaultModels = async (bot: AIBotKey): Promise<string | null> => {
    if (defaultsLoading[bot]) return null
    const baseUrlKey = `${bot}_base_url`
    const keyKey = `${bot}_key`
    const agencyKey = `${bot}_agency`

    const params = new URLSearchParams({ type: bot })
    if (bot === "ollama") {
      const baseUrl = formValues[bot]?.[baseUrlKey]
      if (!baseUrl) {
        modalError(t("errors.baseUrlRequired"))
        return null
      }
      params.set("base_url", baseUrl)
      const keyValue = formValues[bot]?.[keyKey]
      if (keyValue) {
        params.set("key", keyValue)
      }
      const agencyValue = formValues[bot]?.[agencyKey]
      if (agencyValue) {
        params.set("agency", agencyValue)
      }
    }

    setDefaultsLoading((prev) => ({ ...prev, [bot]: true }))
    try {
      const response = await fetch(`/ai/models/list?${params.toString()}`)
      const result = await response.json().catch(() => null)

      if (!response.ok || !result) {
        throw new Error(t("errors.fetchFailed"))
      }

      if (result.code !== 200) {
        throw new Error(result.error || t("errors.fetchFailed"))
      }

      const modelsArray = Array.isArray(result.data?.models) ? result.data.models : []
      if (!modelsArray.length) {
        throw new Error(t("errors.modelsNotFound"))
      }

      const modelsString = modelsArray.join("\n")
      messageSuccess(t("success.fetchSuccess"))
      return modelsString
    } catch (error) {
      modalError(resolveErrorMessage(error, t("errors.fetchFailed")))
      return null
    } finally {
      setDefaultsLoading((prev) => ({ ...prev, [bot]: false }))
    }
  }

  const handleSheetOpenChange = (open: boolean) => {
    if (open && !isAdmin) {
      messageError(t("errors.adminOnly"))
      return
    }
    setSettingsOpenState(open)
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 p-6 sm:p-10">
        <header className="space-y-3">
          <h1 className="text-2xl font-semibold">{t("app.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("app.description")}</p>
        </header>
        <section>
          <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
            {bots.map((bot) => (
              <BotCard
                key={bot.value}
                bot={bot}
                chatLoading={Boolean(chatLoading[bot.value])}
                isAdmin={isAdmin}
                onStartChat={handleStartChat}
                onOpenSettings={handleOpenSettings}
                onShowDescription={handleShowDescription}
              />
            ))}
          </div>
          {!bots.length && (
            <div className="flex items-center justify-center rounded-lg border border-dashed py-16 text-sm text-muted-foreground">
              {t("app.empty")}
            </div>
          )}
        </section>
      </div>
      {isAdmin && (
        <BotSettingsSheet
          open={Boolean(settingsOpen)}
          onOpenChange={handleSheetOpenChange}
          bots={bots}
          activeBot={activeBot}
          onActiveBotChange={handleTabChange}
          fieldMap={fieldMap}
          formValues={formValues}
          initialValues={initialValues}
          loadingMap={settingsLoadingMap}
          savingMap={settingsSavingMap}
          defaultsLoadingMap={defaultsLoading}
          onReload={handleReload}
          onChangeField={handleChangeField}
          onSubmit={handleSubmit}
          onReset={handleReset}
          onUseDefaultModels={handleUseDefaultModels}
          onRegisterModelEditorBackHandler={handleRegisterModelEditorBackHandler}
        />
      )}
    </div>
  )
}

export default App
