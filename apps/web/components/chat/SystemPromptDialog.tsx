'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'

interface SystemPromptDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  value: string
  onSave: (value: string) => void
}

export function SystemPromptDialog({ open, onOpenChange, value, onSave }: SystemPromptDialogProps) {
  const [draft, setDraft] = useState(value)

  const handleOpenChange = (next: boolean) => {
    if (!next) setDraft(value)
    onOpenChange(next)
  }

  const handleSave = () => {
    onSave(draft)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Custom System Prompt</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Override the system prompt for this conversation. This is prepended before soul, agent, and knowledge context.
            </p>
            <textarea
              className="w-full min-h-[120px] text-sm p-2 rounded-md border border-border bg-background resize-y focus:outline-none focus:ring-2 focus:ring-primary/30"
              placeholder="Enter instructions for the AI..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              aria-label="System prompt"
            />
            <div className="flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => handleOpenChange(false)}>Cancel</Button>
              <Button size="sm" onClick={handleSave}>Save</Button>
            </div>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  )
}
