'use client'

import { Button, Dialog, DialogContent, DialogHeader, DialogTitle, Textarea } from '@sloughgpt/strui'
import type { TestModelResult } from '@/hooks/useTestDialog'

interface TestModelDialogProps {
  open: boolean
  prompt: string
  result: TestModelResult | null
  loading: boolean
  onClose: () => void
  onPromptChange: (value: string) => void
  onGenerate: () => void
  onClear: () => void
}

export function TestModelDialog({
  open,
  prompt,
  result,
  loading,
  onClose,
  onPromptChange,
  onGenerate,
  onClear,
}: TestModelDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-base">Test the model</DialogTitle>
        </DialogHeader>
        <Textarea
          value={prompt}
          onChange={e => onPromptChange(e.target.value)}
          placeholder="Type a prompt to test the trained model..."
          rows={3}
          className="text-xs font-mono resize-none"
        />
        <div className="flex gap-2">
          <Button size="sm" onClick={onGenerate} disabled={loading || !prompt.trim()}>
            {loading ? 'Generating...' : 'Generate'}
          </Button>
          <Button size="sm" variant="ghost" onClick={onClear}>
            Clear
          </Button>
        </div>

        {result && (
          <div className="space-y-2">
            {result.error && (
              <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3">
                <p className="text-[10px] text-destructive uppercase tracking-wider mb-1">Error</p>
                <p className="text-xs font-mono text-destructive">{result.error}</p>
              </div>
            )}

            {result.response && (
              <div className="rounded-md border border-border/50 bg-muted/30 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Output</p>
                <p className="text-xs font-mono whitespace-pre-wrap text-foreground">{result.response}</p>
              </div>
            )}

            {(result.model || result.tokens_generated > 0) && (
              <div className="flex gap-3 text-[10px] text-muted-foreground">
                {result.model && <span>Model: {result.model}</span>}
                {result.tokens_generated > 0 && <span>Tokens: {result.tokens_generated}</span>}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
