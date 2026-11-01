'use client'

import { Button, Dialog, DialogContent, DialogHeader, DialogTitle, Textarea, ToggleGroup, ToggleGroupItem } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
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
          <DialogTitle className="text-sm">Test the model</DialogTitle>
        </DialogHeader>
        <Textarea
          aria-label="Test prompt"
          value={prompt}
          onChange={e => onPromptChange(e.target.value)}
          placeholder="Type a prompt to test the trained model..."
          rows={3}
          className="text-[11px] font-mono resize-none"
        />
        <div className="flex items-center gap-2">
          <ToggleGroup
            type="single"
            value={responseFormat}
            onValueChange={(v) => { if (v) onResponseFormatChange(v as 'text' | 'json') }}
            size="sm"
          >
            <ToggleGroupItem value="text" className="text-[10px] h-6 px-1.5">Text</ToggleGroupItem>
            <ToggleGroupItem value="json" className="text-[10px] h-6 px-1.5">JSON</ToggleGroupItem>
          </ToggleGroup>
          <div className="flex gap-1.5 ml-auto">
            <Button size="sm" className="h-7 text-[11px]" onClick={onGenerate} disabled={loading || !prompt.trim()}>
              {loading ? 'Generating...' : 'Generate'}
            </Button>
            <Button size="sm" variant="ghost" className="h-7 text-[11px]" onClick={onClear}>
              Clear
            </Button>
          </div>
        </div>

        {(streaming || showOutput || showError) && (
          <div className="space-y-1.5">
            {showError && (
              <StatusBanner variant="error" message={result!.error || 'An error occurred'} dismissible={false} />
            )}

            {(streaming || showOutput) && (
              <div className="rounded-lg border border-border/50 bg-muted/20 p-2.5">
                <p className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-0.5">
                  Output {streaming && <span className="text-primary animate-pulse">streaming...</span>}
                </p>
                <p className="text-[11px] font-mono whitespace-pre-wrap text-foreground">
                  {displayText}
                  {streaming && <span className="inline-block w-1 h-2.5 bg-primary/70 animate-pulse ml-0.5" />}
                </p>
              </div>
            )}

            {showOutput && (result!.model || result!.tokens_generated > 0) && (
              <div className="flex gap-2.5 text-[9px] text-muted-foreground/60 tabular-nums">
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
