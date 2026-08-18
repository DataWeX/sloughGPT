'use client'

import { useState } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconStar, IconTrash, IconX, IconChevronDown } from '@sloughgpt/strui'
import type { BookmarkedMessage } from '@/features/chat/hooks/useChatBookmarks'

interface ChatBookmarksPanelProps {
  bookmarks: BookmarkedMessage[]
  onRemove: (id: string) => void
  onClear: () => void
  onJumpToMessage?: (id: string) => void
  className?: string
}

export function ChatBookmarksPanel({ bookmarks, onRemove, onClear, onJumpToMessage, className }: ChatBookmarksPanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className={cn('border rounded-lg overflow-hidden', className)}>
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-muted/30 transition-colors"
        aria-expanded={!collapsed}
        aria-controls="bookmarks-panel-content"
      >
        <span className="flex items-center gap-1.5">
          <IconStar className="h-3.5 w-3.5" />
          Bookmarks
          {bookmarks.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-muted">{bookmarks.length}</span>
          )}
        </span>
        <IconChevronDown className={cn('h-3 w-3 transition-transform', collapsed && '-rotate-90')} />
      </button>

      {!collapsed && (
        <div id="bookmarks-panel-content" className="divide-y divide-border/50">
          {bookmarks.length === 0 ? (
            <div className="px-3 py-6 text-center">
              <IconStar className="h-5 w-5 mx-auto mb-1 text-muted-foreground/50" />
              <p className="text-xs text-muted-foreground/60">No bookmarks yet</p>
              <p className="text-[10px] text-muted-foreground/40 mt-0.5">Star a message to save it here</p>
            </div>
          ) : (
            bookmarks.map(bm => (
              <div
                key={bm.id}
                className="group flex items-start gap-2 px-3 py-2 hover:bg-muted/20 transition-colors"
              >
                <button
                  onClick={() => onJumpToMessage?.(bm.id)}
                  className="flex-1 min-w-0 text-left"
                  title={bm.content.slice(0, 120)}
                >
                  <div className={cn(
                    'text-[10px] font-medium mb-0.5',
                    bm.role === 'assistant' ? 'text-primary/70' : 'text-foreground/70'
                  )}>
                    {bm.role === 'assistant' ? 'Assistant' : 'You'}
                    {bm.sessionTitle && <span className="text-muted-foreground/40 ml-1">· {bm.sessionTitle}</span>}
                  </div>
                  <p className="text-[11px] text-muted-foreground line-clamp-2">
                    {bm.content.slice(0, 200)}
                  </p>
                </button>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => onRemove(bm.id)}
                  className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity shrink-0 -mr-1"
                  aria-label="Remove bookmark"
                >
                  <IconX className="h-3 w-3" />
                </Button>
              </div>
            ))
          )}

          {bookmarks.length > 0 && (
            <div className="px-3 py-1.5 border-t border-border/30">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClear}
                className="w-full text-[11px] text-muted-foreground hover:text-error"
              >
                <IconTrash className="h-3 w-3 mr-1" />
                Clear all bookmarks
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
