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

export interface ChatContextValue {
  // Health
  health: ApiHealthSnapshot
  refreshHealth: () => Promise<void>

  // Model
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

  // Souls
  souls: Soul[]
  currentSoul: Soul | null
  setCurrentSoul: (s: Soul | null) => void
  handleSelectSoul: (s: Soul) => void

  // Checkpoints
  checkpoints: Checkpoint[]
  currentCheckpoint: string | undefined
  setCurrentCheckpoint: (c: string | undefined) => void
  onLoadCheckpoint: (name: string) => Promise<void>

  // Agents
  agents: AgentDef[]
  currentAgent: AgentDef | null
  setCurrentAgent: (a: AgentDef | null) => void

  // Vision
  visionCaps: VisionCaps | null
  visionCaptionHistory: string[]
  visionVocabSize: number | undefined

  // Learner
  learnerInfo: LearnerInfo | null
  learnerTraining: boolean
  setLearnerInfo: React.Dispatch<React.SetStateAction<LearnerInfo | null>>
  setLearnerTraining: (t: boolean) => void
  onTrainStep: () => Promise<void>

  // UI callbacks
  onOpenSettings: () => void
  onOpenShortcuts: () => void
  onOpenConversationViewer: () => void

  // Input
  setInput: (value: string | ((prev: string) => string)) => void

  // Toast
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
}

const ChatContext = createContext<ChatContextValue | null>(null)

export function useChatContext() {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChatContext must be used within ChatProvider')
  return ctx
}

interface ChatProviderProps {
  children: ReactNode
  value: ChatContextValue
}

export function ChatProvider({ children, value }: ChatProviderProps) {
  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}
