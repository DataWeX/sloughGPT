'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface Session {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  lastActivity: number
}

interface ChatSessionCloningProps {
  sessions: Session[]
  onClone: (sourceId: string, newTitle: string) => void
  className?: string
}

export const ChatSessionCloning = memo(function ChatSessionCloning({
  sessions,
  onClone,
  className,
}: ChatSessionCloningProps) {
  const [selectedId, setSelectedId] = useState<string>('')
  const [newTitle, setNewTitle] = useState('')
  const [showConfirm, setShowConfirm] = useState(false)

  const selectedSession = sessions.find(s => s.id === selectedId)

  const handleSelect = useCallback((id: string) => {
    setSelectedId(id)
    const session = sessions.find(s => s.id === id)
    if (session) {
      setNewTitle(`${session.title} (Copy)`)
    }
    setShowConfirm(true)
  }, [sessions])

  const handleClone = useCallback(() => {
    if (!selectedId || !newTitle.trim()) return
    onClone(selectedId, newTitle.trim())
    setSelectedId('')
    setNewTitle('')
    setShowConfirm(false)
  }, [selectedId, newTitle, onClone])

  const handleCancel = useCallback(() => {
    setSelectedId('')
    setNewTitle('')
    setShowConfirm(false)
  }, [])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Clone Session</span>
      </div>

      <div className="p-2 space-y-2">
        <p className="text-[10px] text-muted-foreground">
          Select a session to clone with all its messages.
        </p>

        <div className="max-h-[200px] overflow-y-auto space-y-1">
          {sessions.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-2">No sessions</p>
          ) : (
            sessions.map(session => (
              <button
                key={session.id}
                type="button"
                onClick={() => handleSelect(session.id)}
                className={cn(
                  'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                  'hover:bg-muted/50',
                  selectedId === session.id && 'bg-primary/10',
                )}
              >
                <div className="font-medium truncate">{session.title}</div>
                <div className="text-[10px] text-muted-foreground">
                  {session.messages.length} messages
                </div>
              </button>
            ))
          )}
        </div>

        {showConfirm && selectedSession && (
          <div className="border-t pt-2 space-y-2">
            <div className="text-xs text-muted-foreground">
              Cloning: <span className="font-medium">{selectedSession.title}</span>
            </div>
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              placeholder="New session title..."
              className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
              onKeyDown={(e) => e.key === 'Enter' && handleClone()}
              autoFocus
            />
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="text-[10px] h-6"
                onClick={handleClone}
                disabled={!newTitle.trim()}
              >
                <IconCheck className="h-3 w-3 mr-1" />
                Clone
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-[10px] h-6"
                onClick={handleCancel}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
})