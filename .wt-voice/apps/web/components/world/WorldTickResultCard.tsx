'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface TickResult {
  tick: number
  babies: number
}

interface NeuralResult {
  embedding_shape?: number[]
  descriptor?: Record<string, unknown>
}

interface WorldTickResultCardProps {
  tickResult: TickResult | null
  neuralResult?: NeuralResult | null
}

export function WorldTickResultCard({ tickResult, neuralResult }: WorldTickResultCardProps) {
  if (!tickResult && !neuralResult) return null

  return (
    <>
      {tickResult && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Tick Result</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5">
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
              <div className="rounded-lg border-border/40 bg-muted/30 p-2.5 text-center hover:bg-muted/20">
                <div className="text-xs text-muted-foreground">Tick</div>
                <div className="text-[11px] font-mono font-medium tabular-nums">{tickResult.tick}</div>
              </div>
              <div className="rounded-lg border-border/40 bg-muted/30 p-2.5 text-center hover:bg-muted/20">
                <div className="text-xs text-muted-foreground">Babies</div>
                <div className="text-[11px] font-mono font-medium tabular-nums">{tickResult.babies}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {neuralResult && (
        <Card>
          <CardHeader className="pb-2 pt-2.5 px-2.5">
            <CardTitle className="text-[11px] font-medium">Neural Processing</CardTitle>
          </CardHeader>
          <CardContent className="px-2.5 pb-2.5 space-y-3">
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded-lg border-border/40 bg-muted/30 p-2.5 text-center hover:bg-muted/20">
                <div className="text-xs text-muted-foreground">Embedding Shape</div>
                <div className="text-xs font-mono">
                  {neuralResult.embedding_shape ? String(neuralResult.embedding_shape) : 'N/A'}
                </div>
              </div>
            </div>
            {neuralResult.descriptor && (
              <pre className="max-h-[200px] overflow-y-auto rounded bg-muted/30 p-3 text-xs">
                {JSON.stringify(neuralResult.descriptor, null, 2)}
              </pre>
            )}
          </CardContent>
        </Card>
      )}
    </>
  )
}
