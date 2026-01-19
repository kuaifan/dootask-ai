import type { Language, LocalizedText } from "@/lib/i18n-core"
import { getLocalizedText } from "@/lib/i18n-core"

export type AIBotKey =
  | "openai"
  | "claude"
  | "deepseek"
  | "gemini"
  | "grok"
  | "ollama"
  | "zhipu"
  | "qianwen"
  | "wenxin"

export interface AIBotItem {
  value: AIBotKey
  label: string
  src: string
  desc: string
  tags: string[]
  tagLabel?: string
  models?: AIBotModelOption[]
}

export interface AIBotModelOption {
  value: string
  label: string
  support_mcp: boolean
  support_vision: boolean
}

interface AIBotDefinition {
  value: AIBotKey
  label: LocalizedText
  src: string
  desc: LocalizedText
}

const AIBOT_DEFINITIONS: AIBotDefinition[] = [
  {
    value: "openai",
    label: {
      zh: "ChatGPT",
      en: "ChatGPT",
    },
    src: "/ai/ui/avatars/openai.png",
    desc: {
      zh: "我是一个人工智能助手，为用户提供问题解答和指导。我没有具体的身份，只是一个程序。您有什么问题可以问我哦？",
      en: "I am an AI assistant that answers questions and offers guidance. I do not have a real identity—just a program. Feel free to ask me anything!",
    },
  },
  {
    value: "claude",
    label: {
      zh: "Claude",
      en: "Claude",
    },
    src: "/ai/ui/avatars/claude.png",
    desc: {
      zh: "我是 Claude，由 Anthropic 公司创造出来的 AI 助手。我的工作是帮助人类，与人对话并给出解答。",
      en: "I am Claude, an AI assistant created by Anthropic. I help people, hold conversations, and provide answers.",
    },
  },
  {
    value: "deepseek",
    label: {
      zh: "DeepSeek",
      en: "DeepSeek",
    },
    src: "/ai/ui/avatars/deepseek.png",
    desc: {
      zh: "DeepSeek 大语言模型算法是北京深度求索人工智能基础技术研究有限公司推出的深度合成服务算法。",
      en: "DeepSeek is a large language model developed by Beijing DeepSeek. It powers generative AI services and deep synthesis capabilities.",
    },
  },
  {
    value: "gemini",
    label: {
      zh: "Gemini",
      en: "Gemini",
    },
    src: "/ai/ui/avatars/gemini.png",
    desc: {
      zh: "我是由 Google 开发的生成式人工智能助手。它基于同名的 Gemini 系列大型语言模型，是应对 ChatGPT 崛起而开发的。",
      en: "I am Gemini, a generative AI chatbot developed by Google. Built on the Gemini family of large models, I was created in response to the rise of ChatGPT.",
    },
  },
  {
    value: "grok",
    label: {
      zh: "Grok",
      en: "Grok",
    },
    src: "/ai/ui/avatars/grok.png",
    desc: {
      zh: "Grok 是由 xAI 开发的生成式人工智能助手，旨在通过实时回答用户问题来提供帮助。",
      en: "Grok is a generative AI chatbot from xAI designed to help by answering user questions in real time.",
    },
  },
  {
    value: "ollama",
    label: {
      zh: "Ollama",
      en: "Ollama",
    },
    src: "/ai/ui/avatars/ollama.png",
    desc: {
      zh: "Ollama 是一个轻量级、可扩展的框架，旨在让用户能够在本地机器上构建和运行大型语言模型。",
      en: "Ollama is a lightweight, extensible framework that lets you run and build large language models on your local machine.",
    },
  },
  {
    value: "zhipu",
    label: {
      zh: "智谱清言",
      en: "GLM",
    },
    src: "/ai/ui/avatars/zhipu.png",
    desc: {
      zh: "我是智谱清言，是智谱 AI 公司训练的语言模型。我的任务是针对用户的问题和要求提供适当的答复和支持。",
      en: "I am GLM, a language model trained by Zhipu AI. I respond to your questions and provide the support you need.",
    },
  },
  {
    value: "qianwen",
    label: {
      zh: "通义千问",
      en: "Qwen",
    },
    src: "/ai/ui/avatars/qianwen.png",
    desc: {
      zh: "我是达摩院自主研发的超大规模语言模型，能够回答问题、创作文字，还能表达观点、撰写代码。",
      en: "I am Qwen, a large-scale language model from Alibaba's DAMO Academy. I can answer questions, write content, share ideas, and even generate code.",
    },
  },
  {
    value: "wenxin",
    label: {
      zh: "文心一言",
      en: "ERNIE Bot",
    },
    src: "/ai/ui/avatars/wenxin.png",
    desc: {
      zh: "我是文心一言，英文名是 ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。",
      en: "I am ERNIE Bot from Baidu. I can chat, answer questions, assist with creation, and help you obtain information, knowledge, and inspiration efficiently.",
    },
  },
]

export const createLocalizedAIBotList = (lang: Language, previous?: AIBotItem[]): AIBotItem[] => {
  const extras =
    previous?.reduce<
      Partial<Record<AIBotKey, Pick<AIBotItem, "tags" | "tagLabel" | "models">>>
    >((acc, bot) => {
      acc[bot.value] = {
        tags: bot.tags,
        tagLabel: bot.tagLabel,
        models: bot.models,
      }
      return acc
    }, {}) ?? {}

  return AIBOT_DEFINITIONS.map((definition) => {
    const extra = extras[definition.value]
    return {
      value: definition.value,
      label: getLocalizedText(definition.label, lang),
      src: definition.src,
      desc: getLocalizedText(definition.desc, lang),
      tags: extra?.tags ?? [],
      tagLabel: extra?.tagLabel,
      models: extra?.models ?? [],
    }
  })
}
