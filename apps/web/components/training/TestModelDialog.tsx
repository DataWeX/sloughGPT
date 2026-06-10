'use client'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'

interface TestModelDialogProps {
  open: boolean
  prompt: string
  output: string
  loading: boolean
  onClose: () => void
  onPromptChange: (value: string) => void
  onGenerate: () => void
  onClear: () => void
}

export function TestModelDialog({
  open,
  prompt,
  output,
  loading,
  onClose,
  onPromptChange,
  onGenerate,
  onClear,
}: TestModelDialogProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className="bg-background rounded-lg border border-border shadow-xl w-full max-w-lg mx-4 p-5 space-y-4" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">Test the model</h3>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-sm">&times;</button>
        </div>
        <textarea
          value={prompt}
          onChange={e => onPromptChange(e.target.value)}
          placeholder="Type a prompt to test the trained model..."
          rows={3}
          className="w-full rounded-md border border-border/60 bg-background p-2 text-xs font-mono text-foreground resize-none"
        />
        <div className="flex gap-2">
          <Button size="sm" onClick={onGenerate} disabled={loading || !prompt.trim()}>
            {loading ? 'Generating...' : 'Generate'}
          </Button>
          <Button size="sm" variant="ghost" onClick={onClear}>
            Clear
          </Button>
        </div>
        {output && (
          <div className="rounded-md border border-border/50 bg-muted/30 p-3">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Output</p>
            <p className="text-xs font-mono whitespace-pre-wrap text-foreground">{output}</p>
          </div>
        )}
      </div>
    </div>
  )
}
