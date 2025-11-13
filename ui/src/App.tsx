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
import { MCPListCard } from "@/components/aibot/MCPListCard"
import { MCPEditorSheet } from "@/components/aibot/MCPEditorSheet"
import type { AIBotItem, AIBotKey } from "@/data/aibots"
import { createLocalizedAIBotList } from "@/data/aibots"
import { getAISystemConfig, type SystemConfig } from "@/data/aibot-config"
import type { MCPConfig } from "@/data/mcp-config"
import { mergeFields, parseModelNames } from "@/lib/aibot"
import type { GeneratedField } from "@/lib/aibot"
import { useI18n } from "@/lib/i18n-context"
import { loadMCPConfigs, saveMCPConfig, deleteMCPConfig } from "@/lib/mcp-storage"

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

  const [mcps, setMcps] = useState<MCPConfig[]>([])
  const [mcpEditorOpen, setMcpEditorOpen] = useState(false)
  const [editingMcp, setEditingMcp] = useState<MCPConfig | null>(null)

  const settingsOpenRef = useRef(settingsOpen)
  const mcpEditorOpenRef = useRef(mcpEditorOpen)
  const interceptReleaseRef = useRef<(() => void) | null>(null)
  const modelEditorBackHandlerRef = useRef<() => boolean>(() => false)

  const fieldMap = useMemo(() => fieldMapFactory(bots, systemConfig), [bots, systemConfig])

  // 获取所有模型的映射表 (modelValue => modelLabel)
  const allModels = useMemo(() => {
    const models: Record<string, string> = {}
    bots.forEach((bot) => {
      if (bot.tags && bot.tags.length > 0) {
        bot.tags.forEach((tag) => {
          // 从 tags 获取显示名称，使用 tag 作为 key
          // 由于我们没有原始的 model value，这里使用 tag 本身作为 key
          models[tag] = tag
        })
      }
    })
    return models
  }, [bots])

  useEffect(() => {
    settingsOpenRef.current = settingsOpen
  }, [settingsOpen])

  useEffect(() => {
    mcpEditorOpenRef.current = mcpEditorOpen
  }, [mcpEditorOpen])

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
      await loadMcps()
    }

    init().catch((error) => {
      console.error("Failed to initialize AI assistant UI", error)
    })
  }, [])

  const loadMcps = async () => {
    try {
      const configs = await loadMCPConfigs()

      // 检查是否已存在 DooTask MCP
      const hasDooTaskMcp = configs.some(mcp => mcp.isSystem && mcp.name === "DooTask")

      // 如果不存在，添加默认的 DooTask MCP
      if (!hasDooTaskMcp) {
        // 通过 healthz 接口获取 DooTask MCP 状态
        let dootaskEnabled = false
        try {
          const response = await fetch('/apps/mcp_server/healthz')
          if (response.ok) {
            const data = await response.json()
            dootaskEnabled = data.status === "ok"
          }
        } catch (err) {
          console.warn("Failed to fetch DooTask MCP status", err)
        }

        const dootaskMcp: MCPConfig = {
          name: "DooTask",
          config: "{}",
          supportedModels: [],
          enabled: dootaskEnabled,
          isSystem: true
        }

        setMcps([dootaskMcp, ...configs])
      } else {
        setMcps(configs)
      }
    } catch (error) {
      console.error("Failed to load MCP configs", error)
    }
  }

  const refreshBotTags = async () => {
    try {
      const { data } = await requestAPI({
        url: "assistant/models",
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
      console.error("Failed to fetch AI assistant models", error)
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
        if (mcpEditorOpenRef.current) {
          setMcpEditorOpen(false)
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
    if (isAdmin && (settingsOpen || mcpEditorOpen)) {
      void ensureIntercept()
    } else if (!settingsOpen && !mcpEditorOpen) {
      releaseIntercept()
    }
  }, [ensureIntercept, isAdmin, releaseIntercept, settingsOpen, mcpEditorOpen])

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

      // 处理新的 JSON 格式：检查是否是对象数组（包含 id, name, support_mcp）
      let modelsString: string
      if (modelsArray.length > 0 && typeof modelsArray[0] === 'object' && 'id' in modelsArray[0]) {
        // 新格式：{id, name, support_mcp} 转换为 "id|name" 格式
        modelsString = modelsArray
          .map((model: { id: string; name: string; support_mcp: boolean }) =>
            model.name && model.name !== model.id ? `${model.id}|${model.name}` : model.id
          )
          .join("\n")
      } else {
        // 旧格式：直接是字符串数组
        modelsString = modelsArray.join("\n")
      }
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

  const handleAddMcp = () => {
    if (!isAdmin) {
      messageError(t("errors.adminOnly"))
      return
    }
    setEditingMcp(null)
    setMcpEditorOpen(true)
  }

  const handleEditMcp = (mcp: MCPConfig) => {
    if (!isAdmin) {
      messageError(t("errors.adminOnly"))
      return
    }
    setEditingMcp(mcp)
    setMcpEditorOpen(true)
  }

  const handleDeleteMcp = async (name: string) => {
    if (!isAdmin) {
      messageError(t("errors.adminOnly"))
      return
    }
    if (!confirm(t("mcp.deleteMessage"))) {
      return
    }
    try {
      const newMcps = await deleteMCPConfig(name, mcps)
      setMcps(newMcps)
      messageSuccess(t("success.save"))
    } catch (error) {
      messageError(resolveErrorMessage(error, t("errors.submitFailed")))
    }
  }

  const handleSaveMcp = async (mcp: MCPConfig) => {
    try {
      // 如果是系统MCP（DooTask），更新app.state.dootask_mcp
      if (mcp.isSystem && mcp.name === "DooTask") {
        // 只更新本地状态，不保存到文件
        const newMcps = mcps.map(m =>
          m.isSystem && m.name === "DooTask"
            ? { ...m, supportedModels: mcp.supportedModels, enabled: mcp.enabled }
            : m
        )
        setMcps(newMcps)
        messageSuccess(t("success.save"))
      } else {
        const newMcps = await saveMCPConfig(mcp, mcps)
        setMcps(newMcps)
        messageSuccess(t("success.save"))
      }
    } catch (error) {
      messageError(resolveErrorMessage(error, t("errors.submitFailed")))
    }
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
        {isAdmin && (
          <section>
            <MCPListCard
              mcps={mcps}
              bots={bots}
              allModels={allModels}
              onAdd={handleAddMcp}
              onEdit={handleEditMcp}
              onDelete={handleDeleteMcp}
            />
          </section>
        )}
      </div>
      {isAdmin && (
        <>
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
          <MCPEditorSheet
            open={mcpEditorOpen}
            onOpenChange={setMcpEditorOpen}
            mcp={editingMcp}
            bots={bots}
            allModels={allModels}
            onSave={handleSaveMcp}
          />
        </>
      )}
    </div>
  )
}

export default App
