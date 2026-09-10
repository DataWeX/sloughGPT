'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

export interface LoraEvalResult {
  status: string
  elapsed_ms?: number
  report?: string
}

interface LoraEvalHistoryProps {
  history: LoraEvalResult[]
  loading: boolean
}

export function LoraEvalHistory({ history, loading }: LoraEvalHistoryProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Eval History ({history.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <p className="text-xs text-muted-foreground">Loading...</p>
        ) : history.length === 0 ? (
          <p className="text-xs text-muted-foreground">No evaluations yet. Run an eval above.</p>
        ) : (
          <div className="space-y-2 max-h-96 overflow-auto">
            {history.map((h, i) => (
              <div key={i} className="rounded bg-muted/30 px-3 py-2 text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="font-medium">{h.status}</span>
                  {h.elapsed_ms != null && <span className="text-muted-foreground">{h.elapsed_ms}ms</span>}
                </div>
                {h.report && <p className="text-muted-foreground whitespace-pre-wrap">{h.report}</p>}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
