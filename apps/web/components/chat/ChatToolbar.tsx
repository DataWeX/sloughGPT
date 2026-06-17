'use client'

import { memo } from 'react'
import { ConversationsDropdown } from './ConversationsDropdown'
import { ChatSearchBar } from './ChatSearchBar'
import { ModelDropdown } from './ModelDropdown'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { AgentSelectorDropdown } from './AgentSelectorDropdown'
import { LocalEngineToggle } from './LocalEngineToggle'
import { ChatMoreMenu } from './ChatMoreMenu'
import { IconSearch, IconMenu } from '@/components/ui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

const STATUS_COLORS = {
  ok: 'bg-emerald-500',
  degraded: 'bg-amber-500',
  offline: 'bg-red-500',
  loading: 'bg-muted-foreground/40',
} as const

const STATUS_LABELS = {
  ok: 'Model ready',
  degraded: 'Connected, no model loaded',
  offline: 'Server offline',
  loading: 'Checking...',
} as const

export const ChatToolbar = memo(function ChatToolbar() {
  const ctx = useChatToolbarContext()

  return (
    <div className="z-10 flex items-center justify-end lg:justify-center px-2 py-1.5 border-b border-border/30 shrink-0 bg-background/80 backdrop-blur-sm gap-1.5">
      <button
        className="flex lg:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={ctx.sidebar.onToggle}
        aria-label="Toggle conversations"
      >
        <IconMenu className="w-4 h-4" />
      </button>

      <ConversationsDropdown
        conversations={ctx.conversations.conversations}
        currentConversationId={ctx.conversations.sessionIdRef.current}
        onLoadConversation={ctx.conversations.onLoad}
        onStarConversation={ctx.conversations.onStar}
        onPinConversation={ctx.conversations.onPin}
        onNewChat={ctx.conversations.onNewChat}
      />
      <div className={'sm:flex flex-1 sm:flex-initial ' + (ctx.search.showMobile ? 'flex' : 'hidden')}>
        <ChatSearchBar
          searchQuery={ctx.search.query}
          onSearchChange={ctx.search.onChange}
          onClear={ctx.search.onClear}
          matchIndex={ctx.search.matchIndex}
          matchCount={ctx.search.matchCount}
          onPrevMatch={ctx.search.onPrevMatch}
          onNextMatch={ctx.search.onNextMatch}
        />
      </div>

      <button
        className="flex sm:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={() => ctx.search.setShowMobile(!ctx.search.showMobile)}
        aria-label="Toggle search"
      >
        <IconSearch className="w-4 h-4" />
      </button>

      <div className="flex items-center gap-1 sm:gap-1.5">
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${STATUS_COLORS[ctx.health.status]}`}
            title={STATUS_LABELS[ctx.health.status]}
          />
          <ModelDropdown />
        </div>

        <SoulSelectorDropdown
          souls={ctx.soul.souls}
          currentSoul={ctx.soul.current}
          onSelect={ctx.soul.onSelect}
        />

        {ctx.actions.messageCount > 0 && (
          <span className="hidden sm:inline text-[10px] text-muted-foreground tabular-nums" title={`${ctx.actions.messageCount} messages in conversation`}>
            {ctx.actions.messageCount}
          </span>
        )}

        {ctx.knowledge.count > 0 && (
          <div className="relative">
            <button
              onClick={ctx.knowledge.onToggle}
              className="inline-flex items-center gap-1 rounded-full border border-border/50 px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted/50 transition-colors"
              aria-label={`${ctx.knowledge.count} knowledge facts active`}
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5-1.253"/></svg>
              {ctx.knowledge.count}
            </button>
            {ctx.knowledge.showing && (
              <div className="absolute top-full right-0 mt-1 w-80 max-h-64 overflow-y-auto rounded-md border border-border bg-popover p-2 shadow-lg z-50 text-xs text-popover-foreground space-y-1 custom-scrollbar">
                <p className="font-medium text-[10px] text-muted-foreground mb-1">Injected knowledge ({ctx.knowledge.count} facts)</p>
                {ctx.knowledge.context.split('\n').filter(l => l.startsWith('- ')).slice(0, 20).map((l, i) => (
                  <p key={i} className="text-[10px] leading-relaxed text-muted-foreground line-clamp-2">{l.replace(/^- /, '')}</p>
                ))}
              </div>
            )}
          </div>
        )}

        <AgentSelectorDropdown
          agents={ctx.agent.agents}
          currentAgent={ctx.agent.current}
          onSelect={ctx.agent.onSelect}
        />

        <LocalEngineToggle
          visible={!!ctx.localEngine.modelUrl}
          useLocalEngine={ctx.localEngine.useLocal}
          localEngineLoading={ctx.localEngine.loading}
          localArchInfo={ctx.localEngine.archInfo}
          onToggle={ctx.localEngine.onToggle}
        />

        <ChatMoreMenu
          onVoiceMode={ctx.actions.onVoiceMode}
          onToggleTools={ctx.actions.onToggleTools}
          onExportMarkdown={ctx.actions.onExportMarkdown}
          onCopyMarkdown={ctx.actions.onCopyMarkdown}
          onSaveAsDataset={ctx.actions.onSaveAsDataset}
          hasMessages={ctx.actions.hasMessages}
        />
      </div>
    </div>
  )
})
