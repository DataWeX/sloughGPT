'use client'

import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

interface ResultsStepProps {
  checkpoints: UseTrainingCheckpointsReturn
  goToTrain: () => void
}

export function ResultsStep({ checkpoints, goToTrain }: ResultsStepProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">4. Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {checkpoints.checkpoints.length === 0 ? (
          <div className="text-sm text-muted-foreground py-4 text-center">
            No checkpoints yet. Run a training job to see results here.
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              {checkpoints.checkpoints.length} checkpoint(s) saved
            </div>
            <div className="space-y-2">
              {checkpoints.checkpoints.slice(0, 10).map(cp => (
                <div key={cp.name} className="flex items-center justify-between rounded-md border border-border/40 bg-muted/20 px-3 py-2">
                  <div className="min-w-0">
                    <div className="text-xs font-medium truncate">{cp.name}</div>
                    <div className="text-[10px] text-muted-foreground">
                      {cp.loss != null && <span>Loss: {cp.loss.toFixed(4)}</span>}
                      {cp.tags && cp.tags.length > 0 && <span className="ml-2">Tags: {cp.tags.join(', ')}</span>}
                    </div>
                  </div>
                  <Button size="sm" variant="ghost" className="shrink-0" onClick={() => {
                    checkpoints.handleLoadCheckpoint(cp.name, () => {})
                  }}>
                    Load
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 pt-2">
          <Button size="sm" variant="ghost" onClick={goToTrain}>Train more</Button>
        </div>
      </CardContent>
    </Card>
  )
}
