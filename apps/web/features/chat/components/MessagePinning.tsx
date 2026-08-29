'use client'

import { useState, useCallback, useEffect, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface PinnedMessage {
  id: string
  messageId: string
  content: string
  role: string
  pinnedAt: number
  note?: string
}

interface MessagePinningProps {
  messages: ChatMessage[]
  onJumpToMessage: (messageId: string) => void
  className?: string
}

const STORAGE_KEY = 'pinned-messages'

function loadPinned(): PinnedMessage[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function savePinned(pinned: PinnedMessage[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(pinned))
}

export const MessagePinning = memo(function MessagePinning({
  messages,
  onJumpToMessage,
  className,
}: MessagePinningProps) {
  const [pinned, setPinned] = useState<PinnedMessage[]>([])
  const [editingId, setEditingId] = useState<string | null>(null)
  const [noteDraft, setNoteDraft] = useState('')

  useEffect(() => {
    setPinned(loadPinned())
  }, [])

  const handlePin = useCallback((msg: ChatMessage) => {
    if (pinned.some(p => p.messageId === msg.id)) return

    const newPin: PinnedMessage = {
      id: crypto.randomUUID(),
      messageId: msg.id,
      content: msg.content,
      role: msg.role,
      pinnedAt: Date.now(),
    }

    const next = [newPin, ...pinned]
    setPinned(next)
    savePinned(next)
  }, [pinned])

  const handleUnpin = useCallback((id: string) => {
    const next = pinned.filter(p => p.id !== id)
    setPinned(next)
    savePinned(next)
  }, [pinned])

  const handleAddNote = useCallback((id: string) => {
    setEditingId(id)
    const existing = pinned.find(p => p.id === id)
    setNoteDraft(existing?.note || '')
  }, [pinned])

  const handleSaveNote = useCallback(() => {
    if (!editingId) return
    const next = pinned.map(p =>
      p.id === editingId ? { ...p, note: noteDraft.trim() || undefined } : p
    )
    setPinned(next)
    savePinned(next)
    setEditingId(null)
    setNoteDraft('')
  }, [editingId, noteDraft, pinned])

  const handleCancelNote = useCallback(() => {
    setEditingId(null)
    setNoteDraft('')
  }, [])

  const handleClearAll = useCallback(() => {
    setPinned([])
    savePinned([])
  }, [])

  const isPinned = useCallback((messageId: string) => {
    return pinned.some(p => p.messageId === messageId)
  }, [pinned])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">Pinned Messages</span>
          <span className="text-[10px] text-muted-foreground">({pinned.length})</span>
        </div>
        {pinned.length > 0 && (
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handleClearAll}
            title="Clear all pins"
          >
            <IconX className="h-3 w-3" />
          </Button>
        )}
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {pinned.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No pinned messages. Click the pin icon on any message to pin it.
          </p>
        ) : (
          <div className="divide-y">
            {pinned.map(pin => (
              <div key={pin.id} className="px-3 py-2 hover:bg-muted/30 group">
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    className="flex-1 text-left min-w-0"
                    onClick={() => onJumpToMessage(pin.messageId)}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                        {pin.role === 'user' ? 'You' : 'AI'}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {new Date(pin.pinnedAt).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-xs truncate">{pin.content.slice(0, 80)}</p>
                    {pin.note && (
                      <p className="text-[10px] text-muted-foreground mt-1 italic">{pin.note}</p>
                    )}
                  </button>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 shrink-0">
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-5 w-5"
                      onClick={() => handleAddNote(pin.id)}
                      title="Add note"
                    >
                      <IconCheck className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-5 w-5"
                      onClick={() => handleUnpin(pin.id)}
                      title="Unpin"
                    >
                      <IconX className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                {editingId === pin.id && (
                  <div className="mt-2 space-y-1">
                    <input
                      type="text"
                      value={noteDraft}
                      onChange={(e) => setNoteDraft(e.target.value)}
                      placeholder="Add a note..."
                      className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveNote()
                        if (e.key === 'Escape') handleCancelNote()
                      }}
                      autoFocus
                    />
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-[10px] h-5"
                        onClick={handleSaveNote}
                      >
                        Save
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-[10px] h-5"
                        onClick={handleCancelNote}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="px-3 py-2 border-t text-[10px] text-muted-foreground">
        {messages.filter(m => isPinned(m.id)).length} of {messages.length} messages pinned
      </div>
    </div>
  )
})