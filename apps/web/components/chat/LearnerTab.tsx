'use client'

import { Button } from '@sloughgpt/strui'
import { IconBrain, IconRefresh } from '@sloughgpt/strui'
import { LossCurve } from './LossCurve'

interface LearnerInfo {
  total_tokens_ingested: number
  train_steps_completed: number
  current_loss?: number
  loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }>
  n_embed?: number
  n_layer?: number
  arch?: string
}

interface LearnerTabProps {
  learnerInfo: LearnerInfo | null
  learnerTraining: boolean
  onTrainStep: () => Promise<void>
}

export function LearnerTab({ learnerInfo, learnerTraining, onTrainStep }: LearnerTabProps) {
  return (
    <div className="space-y-2">
      {learnerInfo ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 rounded bg-muted/30 border border-border/40">
              <div className="text-[10px] text-muted-foreground">Tokens</div>
              <div className="text-sm font-medium">{learnerInfo.total_tokens_ingested}</div>
            </div>
            <div className="p-2 rounded bg-muted/30 border border-border/40">
              <div className="text-[10px] text-muted-foreground">Steps</div>
              <div className="text-sm font-medium">{learnerInfo.train_steps_completed}</div>
            </div>
          </div>
          {learnerInfo.current_loss != null && (
            <div className="p-2 rounded bg-muted/30 border border-border/40">
              <div className="text-[10px] text-muted-foreground">Current loss</div>
              <div className="text-sm font-medium font-mono">{learnerInfo.current_loss.toFixed(4)}</div>
            </div>
          )}
          {learnerInfo.loss_history && learnerInfo.loss_history.length >= 2 && (
            <LossCurve data={learnerInfo.loss_history} />
          )}
          <Button
            size="sm"
            className="w-full text-xs"
            disabled={learnerTraining}
            onClick={onTrainStep}
          >
            {learnerTraining ? (
              <><IconRefresh className="h-3 w-3 animate-spin mr-1" /> Training...</>
            ) : (
              <><IconBrain className="h-3 w-3 mr-1" /> Train step</>
            )}
          </Button>
        </>
      ) : (
        <div className="space-y-2 animate-pulse">
          <div className="grid grid-cols-2 gap-2">
            <div className="h-10 rounded bg-muted/30 border border-border/40" />
            <div className="h-10 rounded bg-muted/30 border border-border/40" />
          </div>
          <div className="h-10 rounded bg-muted/30 border border-border/40" />
          <div className="h-7 rounded bg-muted/30 border border-border/40" />
        </div>
      )}
    </div>
  )
}
