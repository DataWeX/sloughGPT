'use client'

import { useState, useEffect, useRef } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

interface NoteDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  note: string
  onSave: (note: string) => void
  onDelete?: () => void
}

export function NoteDialog({ open, onOpenChange, note, onSave, onDelete }: NoteDialogProps) {
  const [draft, setDraft] = useState(note)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setDraft(note)
  }, [note, open])

  useEffect(() => {
    if (open) {
      setTimeout(() => textareaRef.current?.focus(), 100)
    }
  }, [open])

  const handleSave = () => {
    onSave(draft.trim())
    onOpenChange(false)
  }

  const handleDelete = () => {
    onDelete?.()
    onOpenChange(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSave()
    }
    if (e.key === 'Escape') {
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{note ? 'Edit Note' : 'Add Note'}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Add a private note to this message. Notes are saved locally and only visible to you.
            </p>
            <textarea
              ref={textareaRef}
              className="w-full min-h-[100px] text-sm p-2 rounded-md border border-border bg-background resize-y focus:outline-none focus:ring-2 focus:ring-primary/30"
              placeholder="Enter your note..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              aria-label="Message note"
            />
            <p className="text-[10px] text-muted-foreground/60">
              Press Ctrl+Enter to save, Escape to cancel
            </p>
            <div className="flex items-center justify-between gap-2">
              <div>
                {note && onDelete && (
                  <Button variant="ghost" size="sm" onClick={handleDelete} className="text-destructive hover:text-destructive">
                    Delete note
                  </Button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>Cancel</Button>
                <Button size="sm" onClick={handleSave}>Save</Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  )
}
