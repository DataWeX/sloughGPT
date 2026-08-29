'use client'

import { useState, useCallback, useMemo, memo } from 'react'
import { Button, IconDownload, IconX, IconRefresh } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface Conversation {
  id: string
  title: string
  messageCount: number
  lastActivity: number
  archived: boolean
  archivedAt?: number
}

interface ConversationArchiveProps {
  conversations: Conversation[]
  onArchive: (id: string) => void
  onRestore: (id: string) => void
  onDelete: (id: string) => void
  className?: string
}

export const ConversationArchive = memo(function ConversationArchive({
  conversations,
  onArchive,
  onRestore,
  onDelete,
  className,
}: ConversationArchiveProps) {
  const [filter, setFilter] = useState<'all' | 'active' | 'archived'>('all')
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const filtered = useMemo(() => {
    switch (filter) {
      case 'active':
        return conversations.filter(c => !c.archived)
      case 'archived':
        return conversations.filter(c => c.archived)
      default:
        return conversations
    }
  }, [conversations, filter])

  const stats = useMemo(() => ({
    total: conversations.length,
    active: conversations.filter(c => !c.archived).length,
    archived: conversations.filter(c => c.archived).length,
  }), [conversations])

  const handleDelete = useCallback((id: string) => {
    if (confirmDelete === id) {
      onDelete(id)
      setConfirmDelete(null)
    } else {
      setConfirmDelete(id)
    }
  }, [confirmDelete, onDelete])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <IconDownload className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs font-medium">Conversation Archive</span>
          <span className="text-[10px] text-muted-foreground">
            ({stats.active} active, {stats.archived} archived)
          </span>
        </div>
      </div>

      <div className="flex gap-1 p-2 border-b">
        {(['all', 'active', 'archived'] as const).map(f => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={cn(
              'text-[10px] px-2 py-1 rounded capitalize transition-colors',
              filter === f ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
            )}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {filtered.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No conversations</p>
        ) : (
          <div className="divide-y">
            {filtered.map(conv => (
              <div key={conv.id} className="flex items-center gap-2 px-3 py-2 hover:bg-muted/30">
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium truncate">{conv.title}</div>
                  <div className="text-[10px] text-muted-foreground">
                    {conv.messageCount} messages · {new Date(conv.lastActivity).toLocaleDateString()}
                    {conv.archived && conv.archivedAt && (
                      <span className="ml-2 text-warning">
                        · Archived {new Date(conv.archivedAt).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  {conv.archived ? (
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-6 w-6"
                      onClick={() => onRestore(conv.id)}
                      title="Restore"
                    >
                      <IconRefresh className="h-3 w-3" />
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-6 w-6"
                      onClick={() => onArchive(conv.id)}
                      title="Archive"
                    >
                      <IconDownload className="h-3 w-3" />
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className={cn(
                      'h-6 w-6',
                      confirmDelete === conv.id && 'text-destructive',
                    )}
                    onClick={() => handleDelete(conv.id)}
                    title={confirmDelete === conv.id ? 'Click again to confirm' : 'Delete'}
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})