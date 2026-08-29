'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ConversationBranchingProps {
  messages: ChatMessage[]
  currentMessageId: string
  onBranch: (branchFromId: string, newMessages: ChatMessage[]) => void
  className?: string
}

interface BranchPreview {
  id: string
  role: string
  content: string
}

export const ConversationBranching = memo(function ConversationBranching({
  messages,
  currentMessageId,
  onBranch,
  className,
}: ConversationBranchingProps) {
  const [open, setOpen] = useState(false)
  const [editContent, setEditContent] = useState('')
  const [previewMessages, setPreviewMessages] = useState<BranchPreview[]>([])

  const currentIndex = messages.findIndex(m => m.id === currentMessageId)

  const handleOpen = useCallback(() => {
    if (currentIndex < 0) return

    const branchMessages = messages.slice(0, currentIndex + 1)
    setPreviewMessages(branchMessages.map(m => ({
      id: m.id,
      role: m.role,
      content: m.content.slice(0, 100),
    })))
    setEditContent('')
    setOpen(true)
  }, [messages, currentIndex])

  const handleBranch = useCallback(() => {
    const trimmed = editContent.trim()
    if (!trimmed || currentIndex < 0) return

    const branchMessages = messages.slice(0, currentIndex + 1)
    const newUserMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmed,
      timestamp: new Date(),
    }

    onBranch(currentMessageId, [...branchMessages, newUserMessage])
    setOpen(false)
  }, [editContent, messages, currentIndex, currentMessageId, onBranch])

  const handleCancel = useCallback(() => {
    setOpen(false)
    setEditContent('')
    setPreviewMessages([])
  }, [])

  if (currentIndex <= 0) return null

  if (!open) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className={cn('text-[10px] h-5 text-muted-foreground/50 hover:text-muted-foreground', className)}
        onClick={handleOpen}
      >
        Branch from here
      </Button>
    )
  }

  return (
    <div className={cn('mt-1 rounded border bg-muted/30 p-2 space-y-2', className)}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-muted-foreground">
          Branch from message {currentIndex + 1} of {messages.length}
        </span>
        <Button variant="ghost" size="icon-sm" className="h-4 w-4" onClick={handleCancel}>
          <IconX className="h-2.5 w-2.5" />
        </Button>
      </div>

      {previewMessages.length > 0 && (
        <div className="max-h-[120px] overflow-y-auto space-y-1">
          {previewMessages.map((msg, i) => (
            <div key={msg.id} className="flex items-start gap-1">
              <span className="text-[10px] text-muted-foreground shrink-0 w-12">
                {msg.role === 'user' ? 'You' : 'AI'}
              </span>
              <span className="text-[10px] text-muted-foreground/70 truncate flex-1">
                {msg.content}
              </span>
            </div>
          ))}
        </div>
      )}

      <textarea
        value={editContent}
        onChange={(e) => setEditContent(e.target.value)}
        placeholder="Enter a different prompt to branch..."
        className="w-full text-xs bg-transparent border rounded px-2 py-1 resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 min-h-[40px]"
        autoFocus
      />

      <div className="flex gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-5"
          onClick={handleBranch}
          disabled={!editContent.trim()}
        >
          <IconCheck className="h-3 w-3 mr-1" />
          Create Branch
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-5"
          onClick={handleCancel}
        >
          Cancel
        </Button>
      </div>
    </div>
  )
})