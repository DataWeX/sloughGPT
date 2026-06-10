'use client'

import { memo } from 'react'
import { ConversationsDropdown } from './ConversationsDropdown'
import { ChatSearchBar } from './ChatSearchBar'
import { ModelDropdown } from './ModelDropdown'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { AgentSelectorDropdown } from './AgentSelectorDropdown'
import { LocalEngineToggle } from './LocalEngineToggle'
import { ChatMoreMenu } from './ChatMoreMenu'
import { IconSearch } from '@/components/ui'
import type { Soul } from '@/lib/souls-controller'
import type { AgentDef } from '@/lib/agents'
import type { Conversation } from '@/lib/session-controller'

type DownloadProgressInfo = {
  percentage: number; status: string; speed_mb_per_sec?: number; eta_seconds?: number;
  bytes_downloaded?: number; total_bytes?: number; current_file?: string; files_completed?: number; files_total?: number;
}

interface ChatToolbarProps {
  sidebarConversations: Conversation[]
  sessionIdRef: React.MutableRefObject<string>
  onLoadSession: (id: string) => Promise<void>
  onStarSession: (id: string, starred: boolean) => Promise<void>
  onPinSession: (id: string, pinned: boolean) => Promise<void>
  onNewChat: () => void
  searchQuery: string
  onSearchChange: (value: string) => void
  onSearchClear: () => void
  matchIndex: number
  matchCount: number
  matchIds: string[]
  onPrevMatch: () => void
  onNextMatch: () => void
  showMobileSearch: boolean
  setShowMobileSearch: (v: boolean) => void
  availableModels: string[]
  model: string
  loadingModel: string | null
  generating: boolean
  modelInfoMap: Record<string, { cached?: boolean; size_gb?: number }>
  downloadProgress: Record<string, DownloadProgressInfo>
  onSelectModel: (model: string) => void
  souls: Soul[]
  currentSoul: Soul | null
  onSelectSoul: (soul: Soul) => void
  knowledgeCtx: { showing: boolean; count: number; context: string }
  onToggleKnowledge: () => void
  agents: AgentDef[]
  currentAgent: AgentDef | null
  onSelectAgent: (agent: AgentDef) => void
  localModelUrl: string
  useLocalEngine: boolean
  localEngineLoading: boolean
  localArchInfo: string | null
  onToggleLocalEngine: () => void
  onVoiceMode: () => void
  onToggleTools: () => void
  onExportMarkdown: () => void
  hasMessages: boolean
}

export const ChatToolbar = memo(function ChatToolbar({
  sidebarConversations, sessionIdRef, onLoadSession, onStarSession, onPinSession, onNewChat,
  searchQuery, onSearchChange, onSearchClear, matchIndex, matchCount, matchIds, onPrevMatch, onNextMatch,
  showMobileSearch, setShowMobileSearch,
  availableModels, model, loadingModel, generating, modelInfoMap, downloadProgress, onSelectModel,
  souls, currentSoul, onSelectSoul,
  knowledgeCtx, onToggleKnowledge,
  agents, currentAgent, onSelectAgent,
  localModelUrl, useLocalEngine, localEngineLoading, localArchInfo, onToggleLocalEngine,
  onVoiceMode, onToggleTools, onExportMarkdown, hasMessages,
}: ChatToolbarProps) {
  return (
    <div className="z-10 flex items-center justify-end lg:justify-center px-2 py-1.5 border-b border-border/30 shrink-0 bg-background/80 backdrop-blur-sm gap-1.5">
      <ConversationsDropdown
        conversations={sidebarConversations}
        currentConversationId={sessionIdRef.current}
        onLoadConversation={onLoadSession}
        onStarConversation={onStarSession}
        onPinConversation={onPinSession}
        onNewChat={onNewChat}
      />
      <div className={'sm:flex flex-1 sm:flex-initial ' + (showMobileSearch ? 'flex' : 'hidden')}>
        <ChatSearchBar
          searchQuery={searchQuery}
          onSearchChange={(value) => { onSearchChange(value) }}
          onClear={onSearchClear}
          matchIndex={matchIndex}
          matchCount={matchCount}
          onPrevMatch={onPrevMatch}
          onNextMatch={onNextMatch}
        />
      </div>

      <button
        className="flex sm:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={() => setShowMobileSearch(!showMobileSearch)}
        aria-label="Toggle search"
      >
        <IconSearch className="w-4 h-4" />
      </button>

      <div className="flex items-center gap-1 sm:gap-1.5">
        <ModelDropdown
          availableModels={availableModels}
          currentModel={model}
          loadingModel={loadingModel}
          generating={generating}
          modelInfoMap={modelInfoMap}
          downloadProgress={downloadProgress}
          onSelectModel={onSelectModel}
        />

        <SoulSelectorDropdown
          souls={souls}
          currentSoul={currentSoul}
          onSelect={onSelectSoul}
        />

        {knowledgeCtx.count > 0 && (
          <div className="relative">
            <button
              onClick={onToggleKnowledge}
              className="inline-flex items-center gap-1 rounded-full border border-border/50 px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted/50 transition-colors"
              aria-label={`${knowledgeCtx.count} knowledge facts active`}
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
              {knowledgeCtx.count}
            </button>
            {knowledgeCtx.showing && (
              <div className="absolute top-full right-0 mt-1 w-80 max-h-64 overflow-y-auto rounded-md border border-border bg-popover p-2 shadow-lg z-50 text-xs text-popover-foreground space-y-1 custom-scrollbar">
                <p className="font-medium text-[10px] text-muted-foreground mb-1">Injected knowledge ({knowledgeCtx.count} facts)</p>
                {knowledgeCtx.context.split('\n').filter(l => l.startsWith('- ')).slice(0, 20).map((l, i) => (
                  <p key={i} className="text-[10px] leading-relaxed text-muted-foreground line-clamp-2">{l.replace(/^- /, '')}</p>
                ))}
              </div>
            )}
          </div>
        )}

        <AgentSelectorDropdown
          agents={agents}
          currentAgent={currentAgent}
          onSelect={onSelectAgent}
        />

        <LocalEngineToggle
          visible={!!localModelUrl}
          useLocalEngine={useLocalEngine}
          localEngineLoading={localEngineLoading}
          localArchInfo={localArchInfo}
          onToggle={onToggleLocalEngine}
        />

        <ChatMoreMenu
          onVoiceMode={onVoiceMode}
          onToggleTools={onToggleTools}
          onExportMarkdown={onExportMarkdown}
          hasMessages={hasMessages}
        />
      </div>
    </div>
  )
})
