'use client'

import { memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { ChatSearchBar } from './ChatSearchBar'
import { ModelDropdown } from './ModelDropdown'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { ChatMoreMenu } from './ChatMoreMenu'
import { IconSearch, IconMenu, IconPlus, IconChat } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/features/chat/contexts/ChatToolbarContext'

export const ChatToolbar = memo(function ChatToolbar() {
  const ctx = useChatToolbarContext()

  const currentName = ctx.conversations.conversations.find(
    c => c.session_id === ctx.conversations.sessionIdRef.current
  )?.name

  return (
    <div className="z-10 flex items-center justify-between px-2 sm:px-3 py-1.5 border-b border-border/30 shrink-0 bg-background/80 backdrop-blur-sm gap-2">
      {/* Left cluster: conversation identity */}
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <button
          type="button"
          className="flex lg:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors shrink-0"
          onClick={ctx.sidebar.onToggle}
          aria-label="Toggle conversations"
        >
          <IconMenu className="w-4 h-4" aria-hidden="true" />
        </button>

        <button
          type="button"
          className="flex items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors text-muted-foreground shrink-0"
          onClick={ctx.conversations.onNewChat}
          aria-label="New chat"
          title="New chat"
        >
          <IconPlus className="w-4 h-4" aria-hidden="true" />
        </button>

        <div className="flex items-center gap-1.5 min-w-0">
          <IconChat className="w-3.5 h-3.5 text-primary/60 shrink-0 hidden sm:block" />
          {currentName ? (
            <span className="text-sm font-medium truncate" title={currentName}>
              {currentName}
            </span>
          ) : (
            <span className="text-sm text-muted-foreground">New chat</span>
          )}
        </div>

        {ctx.actions.hasMessages && (
          <span className="hidden sm:flex items-center gap-1 text-[10px] text-muted-foreground bg-muted/50 px-1.5 py-0.5 rounded shrink-0 tabular-nums">
            <span className="font-mono">{ctx.actions.messageCount}</span>
            <span>msgs</span>
          </span>
        )}
      </div>

      {/* Center: search (collapses to icon on mobile) */}
      <div className={cn('hidden sm:flex items-center', ctx.search.showMobile ? '!flex' : '')}>
        <ChatSearchBar />
      </div>

      <button
        type="button"
        className="flex sm:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors shrink-0"
        onClick={() => ctx.search.setShowMobile(!ctx.search.showMobile)}
        aria-label="Toggle search"
      >
        <IconSearch className="w-4 h-4" aria-hidden="true" />
      </button>

      {/* Right cluster: model + personality + menu */}
      <div className="flex items-center gap-1 sm:gap-1.5 shrink-0">
        <ModelDropdown />
        <SoulSelectorDropdown />
        <ChatMoreMenu />
      </div>
    </div>
  )
})
