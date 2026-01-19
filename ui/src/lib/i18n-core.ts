export type Language = "zh" | "en"

export interface LocalizedText {
  zh: string
  en: string
}

interface TranslationTree {
  [key: string]: string | TranslationTree
}

const TRANSLATIONS: Record<Language, TranslationTree> = {
  zh: {
    app: {
      title: "AI 助手",
      description: "浏览可用的 AI 助手，快速开始对话，并为管理员提供统一的配置入口。",
      empty: "暂无助手配置，请稍后再试。",
    },
    errors: {
      botNotFound: "未找到助手信息",
      botUnavailable: "助手暂未开启",
      loadFailed: "加载失败",
      adminOnly: "仅管理员可配置助手。",
      botUnsupported: "该助手暂不支持配置。",
      submitFailed: "提交失败",
      baseUrlRequired: "请先填写 Base URL",
      fetchFailed: "获取失败",
      modelsNotFound: "未找到默认模型",
    },
    success: {
      save: "修改成功",
      fetchSuccess: "获取成功",
    },
    sheet: {
      title: "AI 设置",
      loading: "配置加载中...",
      empty: "暂无可配置项。",
      reload: "重新加载",
      reset: "重置",
      back: "返回",
      submit: "提交",
      submitting: "提交中...",
      fetching: "获取中...",
      tipPrefix: "获取方式",
      models: {
        column: {
          model: "模型",
          label: "显示名称",
          actions: "操作",
        },
        add: "新增模型",
        empty: "暂无模型，点击下方按钮添加。",
        remove: "删除模型",
        modelPlaceholder: "模型 ID",
        labelPlaceholder: "显示标签（可选）",
        edit: "编辑模型列表",
        drawerTitle: "编辑模型列表",
        drawerDescription: "当前助手：",
        cancel: "取消",
        save: "保存",
      },
    },
    botCard: {
      connecting: "连接中...",
      startChat: "开始聊天",
      settings: "设置",
    },
    mcp: {
      title: "MCP 配置",
      description: "配置模型上下文协议 (Model Context Protocol) 以扩展AI能力",
      name: "MCP 名称",
      namePlaceholder: "请输入MCP名称，例如：filesystem",
      config: "MCP 配置",
      configPlaceholder: '{\n  "command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]\n}',
      configTip: "请输入有效的JSON格式配置",
      configInvalid: "配置格式无效，请输入有效的JSON",
      supportedModels: "支持的模型",
      supportedModelsTip: "选择可以使用此MCP的AI模型",
      enabled: "启用",
      statusEnabled: "已启用",
      statusDisabled: "已禁用",
      addTitle: "添加 MCP",
      editTitle: "编辑 MCP",
      addButton: "添加 MCP",
      edit: "编辑",
      delete: "删除",
      cancel: "取消",
      save: "保存",
      empty: "暂无MCP配置，点击上方按钮添加",
      deleteConfirm: "确认删除",
      deleteMessage: "确定要删除此MCP配置吗？",
    },
    vision: {
      title: "视觉识别",
      description: "配置视觉识别功能以支持图片理解",
      enabled: "启用视觉识别",
      supportedModels: "支持的模型",
      supportedModelsTip: "选择可以接收图片的AI模型",
      maxImageSize: "最大图片尺寸",
      maxImageSizeTip: "像素，宽高中较大值",
      maxFileSize: "最大文件大小",
      maxFileSizeTip: "MB",
      compressionQuality: "压缩质量",
      compressionQualityTip: "1-100，值越大质量越高",
      statusEnabled: "已启用",
      statusDisabled: "已禁用",
      editTitle: "视觉识别配置",
      edit: "编辑",
      cancel: "取消",
      save: "保存",
      imageLimit: "图片限制",
    },
  },
  en: {
    app: {
      title: "AI Assistants",
      description: "Browse the available AI assistants, start chatting quickly, and manage settings in one place.",
      empty: "No assistant configuration yet. Please try again later.",
    },
    errors: {
      botNotFound: "Assistant information not found",
      botUnavailable: "The assistant is not available yet",
      loadFailed: "Failed to load",
      adminOnly: "Only administrators can configure assistants.",
      botUnsupported: "This assistant does not support configuration yet.",
      submitFailed: "Submission failed",
      baseUrlRequired: "Please fill in the Base URL first",
      fetchFailed: "Failed to fetch",
      modelsNotFound: "No default models found",
    },
    success: {
      save: "Saved successfully",
      fetchSuccess: "Fetched successfully",
    },
    sheet: {
      title: "AI Settings",
      loading: "Loading configuration...",
      empty: "No configurable items.",
      reload: "Reload",
      reset: "Reset",
      back: "Back",
      submit: "Submit",
      submitting: "Submitting...",
      fetching: "Fetching...",
      tipPrefix: "How to get",
      models: {
        column: {
          model: "Model",
          label: "Display Label",
          actions: "Actions",
        },
        add: "Add model",
        empty: "No models yet. Use the button below to add one.",
        remove: "Remove model",
        modelPlaceholder: "Model ID",
        labelPlaceholder: "Display label (optional)",
        edit: "Edit Model List",
        drawerTitle: "Edit Model List",
        drawerDescription: "Current assistant:",
        cancel: "Cancel",
        save: "Save",
      },
    },
    botCard: {
      connecting: "Connecting...",
      startChat: "Start Chat",
      settings: "Settings",
    },
    mcp: {
      title: "MCP Configuration",
      description: "Configure Model Context Protocol to extend AI capabilities",
      name: "MCP Name",
      namePlaceholder: "Enter MCP name, e.g.: filesystem",
      config: "MCP Config",
      configPlaceholder: '{\n  "command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]\n}',
      configTip: "Please enter a valid JSON configuration",
      configInvalid: "Invalid configuration format, please enter valid JSON",
      supportedModels: "Supported Models",
      supportedModelsTip: "Select AI models that can use this MCP",
      enabled: "Enabled",
      statusEnabled: "Enabled",
      statusDisabled: "Disabled",
      addTitle: "Add MCP",
      editTitle: "Edit MCP",
      addButton: "Add MCP",
      edit: "Edit",
      delete: "Delete",
      cancel: "Cancel",
      save: "Save",
      empty: "No MCP configurations yet, click the button above to add one",
      deleteConfirm: "Confirm Delete",
      deleteMessage: "Are you sure you want to delete this MCP configuration?",
    },
    vision: {
      title: "Vision Recognition",
      description: "Configure vision recognition to support image understanding",
      enabled: "Enable Vision",
      supportedModels: "Supported Models",
      supportedModelsTip: "Select AI models that can receive images",
      maxImageSize: "Max Image Size",
      maxImageSizeTip: "pixels, max of width/height",
      maxFileSize: "Max File Size",
      maxFileSizeTip: "MB",
      compressionQuality: "Compression Quality",
      compressionQualityTip: "1-100, higher means better quality",
      statusEnabled: "Enabled",
      statusDisabled: "Disabled",
      editTitle: "Vision Configuration",
      edit: "Edit",
      cancel: "Cancel",
      save: "Save",
      imageLimit: "Image Limits",
    },
  },
}

export const FALLBACK_LANGUAGE: Language = "en"
const ZH_LANGUAGE_CODES = new Set(["zh", "zh-cht"])

const translateFromTree = (tree: TranslationTree, keyPath: string[]): string | undefined => {
  let current: string | TranslationTree | undefined = tree
  for (const segment of keyPath) {
    if (!current || typeof current === "string") {
      return undefined
    }
    current = current[segment]
  }
  return typeof current === "string" ? current : undefined
}

export const translateInternal = (lang: Language, key: string): string => {
  const segments = key.split(".")
  const primary = translateFromTree(TRANSLATIONS[lang], segments)
  if (primary) {
    return primary
  }
  if (lang !== FALLBACK_LANGUAGE) {
    const fallback = translateFromTree(TRANSLATIONS[FALLBACK_LANGUAGE], segments)
    if (fallback) {
      return fallback
    }
  }
  return key
}

export const detectLanguage = (): Language => {
  try {
    const params = new URLSearchParams(window.location.search)
    const raw = params.get("lang")
    if (raw) {
      const normalized = raw.trim().toLowerCase()
      if (ZH_LANGUAGE_CODES.has(normalized)) {
        return "zh"
      }
    }
  } catch {
    // ignore errors during detection and fall back to default
  }
  return FALLBACK_LANGUAGE
}

export const getLocalizedText = (localized: LocalizedText, lang: Language) =>
  localized[lang] ?? localized[FALLBACK_LANGUAGE]
