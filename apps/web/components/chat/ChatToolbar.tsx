'use client'

import { memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { ChatSearchBar } from './ChatSearchBar'
import { ModelDropdown } from './ModelDropdown'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { ChatMoreMenu } from './ChatMoreMenu'
import { IconSearch, IconMenu, IconPlus } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

export const ChatToolbar = memo(function ChatToolbar() {
  const ctx = useChatToolbarContext()

  const currentName = ctx.conversations.conversations.find(
    c => c.session_id === ctx.conversations.sessionIdRef.current
  )?.name

  return (
    <div className="z-10 flex items-center justify-end lg:justify-center px-2 py-1.5 border-b border-border/30 shrink-0 bg-background/80 backdrop-blur-sm gap-1.5">
      <button
        className="flex lg:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={ctx.sidebar.onToggle}
        aria-label="Toggle conversations"
      >
        <IconMenu className="w-4 h-4" />
      </button>

      <button
        className="flex lg:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors text-muted-foreground"
        onClick={ctx.conversations.onNewChat}
        aria-label="New chat"
        title="New chat"
      >
        <IconPlus className="w-4 h-4" />
      </button>

      <div className={cn('sm:flex flex-1 sm:flex-initial', ctx.search.showMobile ? 'flex' : 'hidden')}>
        <ChatSearchBar />
      </div>

      <button
        className="flex sm:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={() => ctx.search.setShowMobile(!ctx.search.showMobile)}
        aria-label="Toggle search"
      >
        <IconSearch className="w-4 h-4" />
      </button>

      {currentName && (
        <span className="hidden lg:block text-xs text-muted-foreground truncate max-w-[120px] shrink-0">
          {currentName}
        </span>
      )}

      <div className="flex items-center gap-1 sm:gap-1.5">
        <ModelDropdown />
        <SoulSelectorDropdown />
        <ChatMoreMenu />
      </div>
    </div>
  )
})
