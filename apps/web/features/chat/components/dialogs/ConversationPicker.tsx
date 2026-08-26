'use client'

import { useState, useCallback, useEffect } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconSearch, IconX } from '@sloughgpt/strui'
import { sessionController, type Conversation } from '@/lib/session-controller'
import { truncateMessage } from '@/lib/conversations-utils'

interface ConversationPickerProps {
  open: boolean
  onClose: () => void
  onSelect: (conversationId: string) => void
  currentConversationId?: string
}

export function ConversationPicker({ open, onClose, onSelect, currentConversationId }: ConversationPickerProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    if (!open) return

    setLoading(true)
    sessionController.list()
      .then(sessions => {
        setConversations(sessions.map(s => ({
          id: s.id,
          name: s.name || 'Untitled',
          session_id: s.id,
          created_at: s.created_at,
          updated_at: s.updated_at,
          pinned: s.pinned ?? false,
          starred: s.starred ?? false,
          message_count: s.messages?.length ?? 0,
        })))
      })
      .catch(() => setConversations([]))
      .finally(() => setLoading(false))
  }, [open])

  const filtered = conversations.filter(c =>
    c.id !== currentConversationId &&
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleSelect = useCallback((id: string) => {
    onSelect(id)
    onClose()
  }, [onSelect, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-start justify-center pt-[10vh]">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md bg-background border border-border/50 rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/50">
          <IconSearch className="h-4 w-4 text-muted-foreground/60" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search conversations..."
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/40"
            autoFocus
            aria-label="Search conversations"
          />
          <Button variant="ghost" size="sm" onClick={onClose} className="h-6 px-2 text-xs">
            Esc
          </Button>
        </div>

        <div className="max-h-[300px] overflow-y-auto py-2">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="text-center py-8 px-4">
              <p className="text-sm text-muted-foreground">
                {searchQuery ? 'No conversations match' : 'No other conversations'}
              </p>
            </div>
          )}

          {!loading && filtered.length > 0 && (
            filtered.map((conv) => (
              <button
                key={conv.id}
                type="button"
                onClick={() => handleSelect(conv.id)}
                className="w-full text-left px-4 py-2 hover:bg-muted/40 transition-colors"
              >
                <p className="text-sm font-medium text-foreground line-clamp-1">{conv.name}</p>
                <p className="text-[11px] text-muted-foreground/60 mt-0.5">
                  {conv.message_count} messages · {new Date(conv.updated_at).toLocaleDateString()}
                </p>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
