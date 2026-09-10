'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface VMEditorProps {
  source: string
  onSourceChange: (source: string) => void
  onRun: () => void
  onStep: () => void
  onClear: () => void
  running: boolean
  hasResult: boolean
}

export function VMEditor({
  source,
  onSourceChange,
  onRun,
  onStep,
  onClear,
  running,
  hasResult,
}: VMEditorProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Assembly Source</CardTitle>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={onClear} disabled={!hasResult}>
              Clear
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative border rounded-md overflow-hidden">
          <div className="flex h-80">
            <div className="select-none text-right text-xs text-muted-foreground font-mono bg-muted/20 border-r border-border/50 py-3 px-2 overflow-hidden">
              {source.split('\n').map((_, i) => (
                <div key={i} className="leading-5">
                  {i + 1}
                </div>
              ))}
            </div>
            <textarea
              value={source}
              onChange={(e) => onSourceChange(e.target.value)}
              aria-label="Assembly source code"
              className="flex-1 h-full p-3 font-mono text-sm bg-background resize-none focus:outline-none leading-5"
              spellCheck={false}
              placeholder="[BITS 32]&#10;[ORG 0x1000]&#10;&#10;MOV EAX, 42&#10;HLT"
            />
          </div>
        </div>
        <div className="flex items-center justify-between mt-1">
          <p className="text-xs text-muted-foreground">Ctrl+Enter to run</p>
          <div className="flex gap-1">
            <Button size="sm" onClick={onRun} disabled={running} className="min-w-[80px]">
              {running ? 'Running...' : 'Run'}
            </Button>
            <Button size="sm" variant="outline" onClick={onStep} disabled={running}>
              Step
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
