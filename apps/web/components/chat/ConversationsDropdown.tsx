'use client'

import { useState, useRef, useEffect } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconMessage, IconStar, IconPin, IconChat, IconPlus } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'
import type { Conversation } from '@/lib/session-controller'
import { formatDate, truncateMessage } from '@/lib/conversations-utils'

export function ConversationsDropdown() {
  const ctx = useChatToolbarContext()
  const { conversations, sessionIdRef, onLoad, onStar, onPin, onNewChat } = ctx.conversations
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
      document.addEventListener('keydown', handleEscape)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [open])

  const sorted = [...conversations].sort((a, b) => {
    return new Date(b.updated_at || b.updatedAt || 0).getTime() - new Date(a.updated_at || a.updatedAt || 0).getTime()
  })

  const starred = sorted.filter(c => c.starred).slice(0, 5)
  const recent = sorted.filter(c => !c.starred).slice(0, 8)

  const currentConversationId = sessionIdRef.current
  const handleSelect = (id: string) => {
    onLoad(id)
    setOpen(false)
  }

  return (
    <div className="relative" ref={ref}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen(prev => !prev)}
        className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 h-7 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
        aria-label="Conversations"
        aria-expanded={open}
      >
        <IconMessage className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Conversations</span>
        {conversations.length > 0 && (
          <span className="inline-flex items-center justify-center h-4 min-w-[16px] rounded-full bg-muted px-1 text-[10px] font-medium text-muted-foreground">
            {conversations.length}
          </span>
        )}
      </Button>

      {open && (
        <div className="absolute top-full left-0 mt-1 w-72 sm:w-80 rounded-lg border border-border bg-popover shadow-lg z-50 overflow-hidden">
          <div className="flex items-center justify-between px-3 py-2 border-b border-border/50">
            <span className="text-xs font-medium text-foreground">Conversations</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => { onNewChat(); setOpen(false) }}
              className="text-xs h-6 px-2 gap-1 text-muted-foreground hover:text-foreground"
            >
              <IconPlus className="h-3 w-3" />
              New
            </Button>
          </div>

          <div className="max-h-80 overflow-y-auto overscroll-contain">
            {conversations.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-6 px-3">
                No conversations yet
              </p>
            ) : (
              <>
                {starred.length > 0 && (
                  <div className="px-1.5 pt-1.5">
                    <p className="text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1">
                      Starred
                    </p>
                    <div className="space-y-0.5">
                      {starred.map(c => (
                        <ConvRow
                          key={c.id}
                          conversation={c}
                          isActive={c.id === currentConversationId}
                          onSelect={() => handleSelect(c.id)}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <div className="px-1.5 pt-1.5 pb-1.5">
                  {starred.length > 0 && (
                    <p className="text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider px-2 py-1">
                      Recent
                    </p>
                  )}
                  <div className="space-y-0.5">
                    {recent.map(c => (
                      <ConvRow
                        key={c.id}
                        conversation={c}
                        isActive={c.id === currentConversationId}
                        onSelect={() => handleSelect(c.id)}
                      />
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function formatDateShort(date: string | Date | undefined): string {
  if (!date) return ''
  const d = new Date(date)
  if (isNaN(d.getTime())) return ''
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function ConvRow({
  conversation: c,
  isActive,
  onSelect,
}: {
  conversation: Conversation
  isActive: boolean
  onSelect: () => void
}) {
  const msgCount = c.messages?.length ?? c.message_count ?? 0
  const lastMsg = c.messages?.[c.messages.length - 1]?.content || ''

  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md px-2 py-1.5 cursor-pointer transition-colors",
        isActive
          ? "bg-primary/10"
          : "hover:bg-muted/40"
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); onSelect() } }}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          {c.pinned && <IconPin className="h-2.5 w-2.5 text-primary shrink-0" />}
          {c.starred && <IconStar className="h-2.5 w-2.5 text-warning shrink-0" filled />}
          <p className="text-xs font-medium truncate text-foreground">{c.name}</p>
        </div>
        {lastMsg && (
          <p className="text-[11px] text-muted-foreground/70 mt-0.5 line-clamp-1">
            {truncateMessage(lastMsg, 36)}
          </p>
        )}
        <div className="flex items-center gap-1 mt-0.5">
          <IconChat className="h-2.5 w-2.5 text-muted-foreground/50 shrink-0" />
          <span className="text-[10px] text-muted-foreground/50">
            {msgCount} msgs · {formatDateShort(c.updated_at || c.updatedAt)}
          </span>
        </div>
      </div>
    </div>
  )
}
