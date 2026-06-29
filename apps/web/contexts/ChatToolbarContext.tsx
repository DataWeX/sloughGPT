'use client'

import { createContext, useContext, type ReactNode } from 'react'
import type { Soul } from '@/lib/souls-controller'
import type { AgentDef } from '@/lib/agents'
import type { Conversation } from '@/lib/session-controller'

type DownloadProgressInfo = {
  percentage: number; status: string; speed_mb_per_sec?: number; eta_seconds?: number;
  bytes_downloaded?: number; total_bytes?: number; current_file?: string; files_completed?: number; files_total?: number;
}

interface ConversationsGroup {
  conversations: Conversation[]
  sessionIdRef: React.MutableRefObject<string>
  onLoad: (id: string) => Promise<void>
  onStar: (id: string, starred: boolean) => Promise<void>
  onPin: (id: string, pinned: boolean) => Promise<void>
  onNewChat: () => void
}

interface SearchGroup {
  query: string
  onChange: (value: string) => void
  onClear: () => void
  matchIndex: number
  matchCount: number
  matchIds: string[]
  onPrevMatch: () => void
  onNextMatch: () => void
  showMobile: boolean
  setShowMobile: (v: boolean) => void
  searchInputRef?: React.RefObject<HTMLInputElement | null>
}

interface ModelGroup {
  availableModels: string[]
  current: string
  loading: string | null
  generating: boolean
  infoMap: Record<string, { cached?: boolean; size_gb?: number }>
  descriptions?: Record<string, string>
  downloadProgress: Record<string, DownloadProgressInfo>
  onSelect: (model: string) => void
  onUnload?: () => void
}

interface SoulGroup {
  souls: Soul[]
  current: Soul | null
  onSelect: (soul: Soul) => void
}

interface KnowledgeGroup {
  showing: boolean
  count: number
  context: string
  onToggle: () => void
}

interface AgentGroup {
  agents: AgentDef[]
  current: AgentDef | null
  onSelect: (agent: AgentDef) => void
}

interface LocalEngineGroup {
  modelUrl: string
  useLocal: boolean
  loading: boolean
  archInfo: string | null
  onToggle: () => void
}

interface ActionsGroup {
  onVoiceMode: () => void
  onToggleTools: () => void
  onExportMarkdown: () => void
  onCopyMarkdown?: () => void
  onSaveAsDataset?: () => void
  onSystemPrompt: () => void
  onSearchConversations: () => void
  hasMessages: boolean
  messageCount: number
  bookmarkCount: number
}

interface SidebarGroup {
  open: boolean
  onToggle: () => void
  onClose: () => void
}

interface HealthGroup {
  status: 'ok' | 'degraded' | 'offline' | 'loading'
  summary: string
  modelLoaded: boolean
  modelType: string
}

export interface ChatToolbarContextValue {
  conversations: ConversationsGroup
  search: SearchGroup
  model: ModelGroup
  soul: SoulGroup
  knowledge: KnowledgeGroup
  agent: AgentGroup
  localEngine: LocalEngineGroup
  actions: ActionsGroup
  health: HealthGroup
  sidebar: SidebarGroup
}

const ChatToolbarContext = createContext<ChatToolbarContextValue | null>(null)

export function useChatToolbarContext() {
  const ctx = useContext(ChatToolbarContext)
  if (!ctx) throw new Error('useChatToolbarContext must be used within ChatToolbarProvider')
  return ctx
}

interface ChatToolbarProviderProps {
  children: ReactNode
  value: ChatToolbarContextValue
}

export function ChatToolbarProvider({ children, value }: ChatToolbarProviderProps) {
  return (
    <ChatToolbarContext.Provider value={value}>
      {children}
    </ChatToolbarContext.Provider>
  )
}
