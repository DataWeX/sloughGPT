'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'

interface EvalRunCardProps {
  adapterPath: string
  soul: string
  running: boolean
  onAdapterPathChange: (value: string) => void
  onSoulChange: (value: string) => void
  onRun: () => void
}

export function EvalRunCard({
  adapterPath,
  soul,
  running,
  onAdapterPathChange,
  onSoulChange,
  onRun,
}: EvalRunCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Run Evaluation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Adapter Path</label>
            <Input
              value={adapterPath}
              onChange={e => onAdapterPathChange(e.target.value)}
              className="h-8 text-xs font-mono mt-1"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Soul</label>
            <Input
              value={soul}
              onChange={e => onSoulChange(e.target.value)}
              className="h-8 text-xs mt-1"
            />
          </div>
        </div>
        <Button size="sm" onClick={onRun} disabled={running}>
          {running ? 'Running...' : 'Run Eval'}
        </Button>
      </CardContent>
    </Card>
  )
}
