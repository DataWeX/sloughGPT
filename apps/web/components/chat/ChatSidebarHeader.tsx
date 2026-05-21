'use client'

import { Button } from '@/components/ui/button'
import { IconChevronRight, IconPlus, IconStar } from '@/components/ui'
import { SearchInput } from '@/components/ui/input'
import { cn } from '@/lib/cn'

interface ChatSidebarHeaderProps {
  searchQuery: string
  onSearchChange: (value: string) => void
  onNewChat?: () => void
  collapsed?: boolean
  onToggleCollapse?: () => void
  starredCount: number
  className?: string
  onOpenConversationSearch?: () => void
}

export function ChatSidebarHeader({
  searchQuery,
  onSearchChange,
  onNewChat,
  collapsed,
  onToggleCollapse,
  starredCount,
  onOpenConversationSearch,
  className,
}: ChatSidebarHeaderProps) {
  if (collapsed) {
    return (
      <div className="flex flex-col items-center gap-2 px-2 py-3 border-b shrink-0">
        {onNewChat && (
          <Button variant="ghost" size="sm" onClick={onNewChat} className="h-8 w-8 p-0" aria-label="New chat">
            <IconPlus className="h-3 w-3" />
          </Button>
        )}
        {onToggleCollapse && (
          <Button variant="ghost" size="sm" onClick={onToggleCollapse} className="h-8 w-8 p-0" aria-label="Expand sidebar">
            <IconChevronRight className="h-4 w-4" />
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className={cn("px-3 py-2 border-b bg-muted/30 shrink-0", className)}>
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center gap-1.5 flex-1">
          <span className="font-semibold text-sm tracking-tight">Chats</span>
          {starredCount > 0 && (
            <span className="inline-flex items-center gap-0.5 text-xs text-warning bg-warning/10 px-1.5 py-0.5 rounded-full">
              <IconStar className="h-2 w-2" filled />
              {starredCount}
            </span>
          )}
        </div>
        {onNewChat && (
          <Button size="sm" onClick={onNewChat} className="h-5 px-1.5 text-xs" aria-label="New chat">
            <IconPlus className="h-3 w-3" />
          </Button>
        )}
      </div>
      <SearchInput
        value={searchQuery}
        onChange={onSearchChange}
        placeholder="Search..."
        className="text-xs"
      />
      {onOpenConversationSearch && (
        <button
          onClick={onOpenConversationSearch}
          className="mt-1 w-full text-left text-[10px] text-muted-foreground/60 hover:text-foreground transition-colors px-1 py-0.5 rounded hover:bg-muted/30"
        >
          Search all conversations…
        </button>
      )}
    </div>
  )
}
