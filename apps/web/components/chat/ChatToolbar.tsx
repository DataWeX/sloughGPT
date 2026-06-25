'use client'

import { memo } from 'react'
import { ConversationsDropdown } from './ConversationsDropdown'
import { ChatSearchBar } from './ChatSearchBar'
import { ModelDropdown } from './ModelDropdown'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { ChatMoreMenu } from './ChatMoreMenu'
import { IconSearch, IconMenu } from '@/components/ui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

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

      <button
        className="flex lg:hidden items-center justify-center h-6 w-6 rounded-md hover:bg-muted/60 transition-colors text-muted-foreground"
        onClick={ctx.conversations.onNewChat}
        aria-label="New chat"
        title="New chat"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4"/></svg>
      </button>

      <ConversationsDropdown />
      <div className={'sm:flex flex-1 sm:flex-initial ' + (ctx.search.showMobile ? 'flex' : 'hidden')}>
        <ChatSearchBar />
      </div>

      <button
        className="flex sm:hidden items-center justify-center h-7 w-7 rounded-md hover:bg-muted/60 transition-colors"
        onClick={() => ctx.search.setShowMobile(!ctx.search.showMobile)}
        aria-label="Toggle search"
      >
        <IconSearch className="w-4 h-4" />
      </button>

      <div className="flex items-center gap-1 sm:gap-1.5">
        <ModelDropdown />
        <SoulSelectorDropdown />
        <ChatMoreMenu />
      </div>
    </div>
  )
})
