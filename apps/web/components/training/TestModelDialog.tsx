'use client'

import { Button, Dialog, DialogContent, DialogHeader, DialogTitle, Textarea, ToggleGroup, ToggleGroupItem } from '@sloughgpt/strui'
import type { TestModelResult } from '@/hooks/useTestDialog'

interface TestModelDialogProps {
  open: boolean
  prompt: string
  result: TestModelResult | null
  loading: boolean
  streaming: boolean
  streamingText: string
  responseFormat: 'text' | 'json'
  onClose: () => void
  onPromptChange: (value: string) => void
  onGenerate: () => void
  onClear: () => void
  onResponseFormatChange: (format: 'text' | 'json') => void
}

export function TestModelDialog({
  open,
  prompt,
  result,
  loading,
  streaming,
  streamingText,
  responseFormat,
  onClose,
  onPromptChange,
  onGenerate,
  onClear,
  onResponseFormatChange,
}: TestModelDialogProps) {
  const displayText = streaming ? streamingText : result?.response || ''
  const showError = !streaming && result?.error
  const showOutput = !streaming && result?.response

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose() }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-base">Test the model</DialogTitle>
        </DialogHeader>
        <Textarea
          aria-label="Test prompt"
          value={prompt}
          onChange={e => onPromptChange(e.target.value)}
          placeholder="Type a prompt to test the trained model..."
          rows={3}
          className="text-xs font-mono resize-none"
        />
        <div className="flex items-center gap-3">
          <ToggleGroup
            type="single"
            value={responseFormat}
            onValueChange={(v) => { if (v) onResponseFormatChange(v as 'text' | 'json') }}
            size="sm"
          >
            <ToggleGroupItem value="text" className="text-xs h-7 px-2">Text</ToggleGroupItem>
            <ToggleGroupItem value="json" className="text-xs h-7 px-2">JSON</ToggleGroupItem>
          </ToggleGroup>
          <div className="flex gap-2 ml-auto">
            <Button size="sm" onClick={onGenerate} disabled={loading || !prompt.trim()}>
              {loading ? 'Generating...' : 'Generate'}
            </Button>
            <Button size="sm" variant="ghost" onClick={onClear}>
              Clear
            </Button>
          </div>
        </div>

        {(streaming || showOutput || showError) && (
          <div className="space-y-2">
            {showError && (
              <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3">
                <p className="text-[10px] text-destructive uppercase tracking-wider mb-1">Error</p>
                <p className="text-xs font-mono text-destructive">{result!.error}</p>
              </div>
            )}

            {(streaming || showOutput) && (
              <div className="rounded-md border border-border/50 bg-muted/30 p-3">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                  Output {streaming && <span className="text-primary animate-pulse">streaming...</span>}
                </p>
                <p className="text-xs font-mono whitespace-pre-wrap text-foreground">
                  {displayText}
                  {streaming && <span className="inline-block w-1.5 h-3 bg-primary/70 animate-pulse ml-0.5" />}
                </p>
              </div>
            )}

            {showOutput && (result!.model || result!.tokens_generated > 0) && (
              <div className="flex gap-3 text-[10px] text-muted-foreground">
                {result!.model && <span>Model: {result!.model}</span>}
                {result!.tokens_generated > 0 && <span>Tokens: {result!.tokens_generated}</span>}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
