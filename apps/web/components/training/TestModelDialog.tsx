'use client'

import { Button } from '@sloughgpt/strui'
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
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div className="bg-background rounded-lg border border-border shadow-xl w-full max-w-lg mx-4 p-5 space-y-4" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">Test the model</h3>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-sm" aria-label="Close">&times;</button>
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

        {result && (
          <div className="space-y-2">
            {result.error && (
              <div className="rounded-md border border-red-500/30 bg-red-500/5 p-3">
                <p className="text-[10px] text-red-500 uppercase tracking-wider mb-1">Error</p>
                <p className="text-xs font-mono text-red-500">{result.error}</p>
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
      </div>
    </div>
  )
}
