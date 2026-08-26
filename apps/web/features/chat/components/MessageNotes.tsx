'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconEdit, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { useMessageNotes, type MessageNote } from '@/features/chat/hooks/useMessageNotes'

interface MessageNotesProps {
  sessionId: string | null
  messageId: string
  className?: string
}

export const MessageNotes = memo(function MessageNotes({
  sessionId,
  messageId,
  className,
}: MessageNotesProps) {
  const { getNote, setNote, removeNote, hasNote } = useMessageNotes({ sessionId })
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  const existing = getNote(messageId)

  const handleStartEdit = useCallback(() => {
    setDraft(existing || '')
    setEditing(true)
  }, [existing])

  const handleSave = useCallback(async () => {
    const trimmed = draft.trim()
    if (trimmed) {
      await setNote(messageId, trimmed)
    } else if (existing) {
      await removeNote(messageId)
    }
    setEditing(false)
  }, [draft, messageId, existing, setNote, removeNote])

  const handleCancel = useCallback(() => {
    setEditing(false)
    setDraft('')
  }, [])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSave()
    } else if (e.key === 'Escape') {
      handleCancel()
    }
  }, [handleSave, handleCancel])

  if (!sessionId) return null

  if (editing) {
    return (
      <div className={cn('mt-1 rounded border bg-muted/30 p-2', className)}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Add a note..."
          className="w-full text-xs bg-transparent border-0 p-0 resize-none focus:outline-none focus:ring-0 placeholder:text-muted-foreground/50 min-h-[60px]"
          autoFocus
        />
        <div className="flex items-center gap-1 mt-1">
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handleSave}
            title="Save (Ctrl+Enter)"
          >
            <IconCheck className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handleCancel}
            title="Cancel (Esc)"
          >
            <IconX className="h-3 w-3" />
          </Button>
          <span className="text-[10px] text-muted-foreground ml-1">
            Ctrl+Enter to save
          </span>
        </div>
      </div>
    )
  }

  if (existing) {
    return (
      <div className={cn('mt-1 rounded border bg-primary/5 p-2 group', className)}>
        <div className="flex items-start justify-between gap-2">
          <p className="text-xs text-muted-foreground flex-1 whitespace-pre-wrap">{existing}</p>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5 opacity-0 group-hover:opacity-100 shrink-0"
            onClick={handleStartEdit}
            title="Edit note"
          >
            <IconEdit className="h-3 w-3" />
          </Button>
        </div>
      </div>
    )
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className={cn('text-[10px] h-5 text-muted-foreground/50 hover:text-muted-foreground', className)}
      onClick={handleStartEdit}
    >
      <IconEdit className="h-3 w-3 mr-1" />
      Add note
    </Button>
  )
})