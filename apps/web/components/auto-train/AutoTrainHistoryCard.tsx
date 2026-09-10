'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'

interface LastTrain {
  started_at: string
  completed_at: string | null
  pairs_used: number
  checkpoint: string
}

interface AutoTrainHistoryCardProps {
  lastTrain: LastTrain | null
}

export function AutoTrainHistoryCard({ lastTrain }: AutoTrainHistoryCardProps) {
  return (
    <Card data-testid="auto-train-history">
      <CardHeader>
        <CardTitle className="text-base">Last Training Run</CardTitle>
      </CardHeader>
      <CardContent>
        {!lastTrain ? (
          <div className="text-sm text-muted-foreground text-center py-4">No training runs yet.</div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Started</span>
              <span className="text-xs">{new Date(lastTrain.started_at).toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Completed</span>
              <span className="text-xs">
                {lastTrain.completed_at
                  ? new Date(lastTrain.completed_at).toLocaleString()
                  : <span className="text-warning">In progress...</span>
                }
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Pairs Used</span>
              <span className="text-xs font-medium">{lastTrain.pairs_used}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Checkpoint</span>
              <span className="text-[10px] font-mono bg-muted px-1.5 py-0.5 rounded">{lastTrain.checkpoint}</span>
            </div>
            {lastTrain.completed_at && (
              <div className="text-[10px] text-muted-foreground text-right">
                Duration: {Math.round((new Date(lastTrain.completed_at).getTime() - new Date(lastTrain.started_at).getTime()) / 1000)}s
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
