'use client'

import { createContext, useContext, type ReactNode } from 'react'
import type { AgentDef } from '@/lib/agents'
import type { Soul } from '@/lib/souls-controller'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'

interface LearnerInfo {
  total_tokens_ingested: number
  train_steps_completed: number
  current_loss: number | undefined
  loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }>
  n_embed?: number
  n_layer?: number
  arch?: string
}

interface Checkpoint {
  name: string
  loss?: number
  traits?: string[]
  is_loaded?: boolean
  eval_verdict?: string
}

interface VisionCaps {
  images_learned?: number
  trained?: boolean
  status?: string
}

// ── Sub-context 1: Health (changes on health poll) ────────────────────────

export interface ChatHealthContextValue {
  health: ApiHealthSnapshot
  refreshHealth: () => Promise<void>
}

const ChatHealthContext = createContext<ChatHealthContextValue | null>(null)

export function useChatHealth() {
  const ctx = useContext(ChatHealthContext)
  if (!ctx) throw new Error('useChatHealth must be used within ChatProvider')
  return ctx
}

// ── Sub-context 2: Model state (changes on user action) ───────────────────

export interface ChatModelContextValue {
  model: string
  setModel: (m: string) => void
  availableModels: string[]
  modelInfoMap: Record<string, { cached?: boolean; size_gb?: number }>
  temperature: number
  setTemperature: (t: number) => void
  maxTokens: number
  setMaxTokens: (t: number) => void
  loadingModel: string | null
  handleSelectModel: (m: string) => Promise<void>
  handleUnloadModel: () => Promise<void>
  souls: Soul[]
  currentSoul: Soul | null
  setCurrentSoul: (s: Soul | null) => void
  handleSelectSoul: (s: Soul) => void
  checkpoints: Checkpoint[]
  currentCheckpoint: string | undefined
  setCurrentCheckpoint: (c: string | undefined) => void
  onLoadCheckpoint: (name: string) => Promise<void>
  agents: AgentDef[]
  currentAgent: AgentDef | null
  setCurrentAgent: (a: AgentDef | null) => void
  visionCaps: VisionCaps | null
  visionCaptionHistory: string[]
  visionVocabSize: number | undefined
  learnerInfo: LearnerInfo | null
  learnerTraining: boolean
  setLearnerInfo: React.Dispatch<React.SetStateAction<LearnerInfo | null>>
  setLearnerTraining: (t: boolean) => void
  onTrainStep: () => Promise<void>
  setInput: (value: string | ((prev: string) => string)) => void
}

const ChatModelContext = createContext<ChatModelContextValue | null>(null)

export function useChatModel() {
  const ctx = useContext(ChatModelContext)
  if (!ctx) throw new Error('useChatModel must be used within ChatProvider')
  return ctx
}

// ── Sub-context 3: UI callbacks (stable references) ───────────────────────

export interface ChatUICallbacksContextValue {
  onOpenSettings: () => void
  onOpenShortcuts: () => void
  onOpenConversationViewer: () => void
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
}

const ChatUICallbacksContext = createContext<ChatUICallbacksContextValue | null>(null)

export function useChatUI() {
  const ctx = useContext(ChatUICallbacksContext)
  if (!ctx) throw new Error('useChatUI must be used within ChatProvider')
  return ctx
}

// ── Combined type for backward compat ─────────────────────────────────────

export interface ChatContextValue extends ChatHealthContextValue, ChatModelContextValue, ChatUICallbacksContextValue {}

// ── Backward-compat hook ──────────────────────────────────────────────────

export function useChatContext(): ChatContextValue {
  const health = useChatHealth()
  const model = useChatModel()
  const ui = useChatUI()
  return { ...health, ...model, ...ui }
}

// ── Provider ──────────────────────────────────────────────────────────────

interface ChatProviderProps {
  children: ReactNode
  health: ChatHealthContextValue
  model: ChatModelContextValue
  ui: ChatUICallbacksContextValue
}

export function ChatProvider({ children, health, model, ui }: ChatProviderProps) {
  return (
    <ChatHealthContext.Provider value={health}>
      <ChatModelContext.Provider value={model}>
        <ChatUICallbacksContext.Provider value={ui}>
          {children}
        </ChatUICallbacksContext.Provider>
      </ChatModelContext.Provider>
    </ChatHealthContext.Provider>
  )
}
