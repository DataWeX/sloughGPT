'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { IconPlus, IconStar, IconPin, IconChat, IconChevronRight, IconX } from '@/components/ui'
import { cn } from '@/lib/cn'
import type { Conversation } from '@/lib/session-controller'

interface ConversationSidebarProps {
  conversations: Conversation[]
  currentConversationId?: string
  onLoadConversation: (id: string) => void
  onNewChat: () => void
  open: boolean
  onClose: () => void
}

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 7) return `${diffDays}d`
  return date.toLocaleDateString()
}

function truncateMessage(content: string, maxLen = 36): string {
  if (!content) return 'Empty conversation'
  const firstLine = content.split('\n')[0]
  return firstLine.length > maxLen ? firstLine.slice(0, maxLen) + '…' : firstLine
}

function SidebarContent({
  conversations,
  currentConversationId,
  onLoadConversation,
  onNewChat,
  onClose,
  isDrawer,
}: {
  conversations: Conversation[]
  currentConversationId?: string
  onLoadConversation: (id: string) => void
  onNewChat: () => void
  onClose?: () => void
  isDrawer?: boolean
}) {
  const sorted = [...conversations].sort((a, b) => {
    return new Date(b.updated_at || b.updatedAt || 0).getTime() - new Date(a.updated_at || a.updatedAt || 0).getTime()
  })

  const starred = sorted.filter(c => c.starred).slice(0, 10)
  const recent = sorted.filter(c => !c.starred).slice(0, 20)

  const handleSelect = (id: string) => {
    onLoadConversation(id)
    onClose?.()
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
        <span className="text-xs font-medium text-foreground">Conversations</span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { onNewChat(); onClose?.() }}
            className="text-xs h-6 px-2 gap-1 text-muted-foreground hover:text-foreground"
          >
            <IconPlus className="h-3 w-3" />
            New
          </Button>
          {isDrawer && onClose && (
            <button
              onClick={onClose}
              className="flex items-center justify-center h-6 w-6 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close sidebar"
            >
              <IconX className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto overscroll-contain px-1.5 py-1">
        {conversations.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-8 px-3">
            No conversations yet. Click + New to start.
          </p>
        ) : (
          <>
            {starred.length > 0 && (
              <div className="mb-1">
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

            <div>
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

      <Link
        href="/conversations"
        onClick={onClose}
        className="flex items-center gap-1 px-3 py-2 border-t border-border/50 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-colors shrink-0"
      >
        View all conversations
        <IconChevronRight className="h-3 w-3" />
      </Link>
    </div>
  )
}

export function ConversationSidebar(props: ConversationSidebarProps) {
  const { open, onClose } = props

  return (
    <>
      {/* Desktop: always visible */}
      <aside className="hidden lg:flex w-64 shrink-0 flex-col border-r border-border/40 bg-background/80">
        <SidebarContent {...props} isDrawer={false} />
      </aside>

      {/* Mobile: drawer overlay */}
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={onClose}
          />
          <aside className="absolute inset-y-0 left-0 w-72 flex flex-col bg-background shadow-xl">
            <SidebarContent {...props} isDrawer={true} />
          </aside>
        </div>
      )}
    </>
  )
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
        isActive ? "bg-primary/10" : "hover:bg-muted/40"
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
            {truncateMessage(lastMsg)}
          </p>
        )}
        <div className="flex items-center gap-1 mt-0.5">
          <IconChat className="h-2.5 w-2.5 text-muted-foreground/50 shrink-0" />
          <span className="text-[10px] text-muted-foreground/50">
            {msgCount} msgs · {formatDate(c.updated_at || c.updatedAt)}
          </span>
        </div>
      </div>
    </div>
  )
}
