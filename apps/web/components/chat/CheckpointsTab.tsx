'use client'

import { IconCheck } from '@/components/ui'
import { ModelDropdown } from './ModelDropdown'
import { cn } from '@/lib/cn'

interface Checkpoint {
  name: string
  loss?: number
  traits?: string[]
  is_loaded?: boolean
  eval_verdict?: string
}

interface CheckpointsTabProps {
  checkpoints: Checkpoint[]
  onLoadCheckpoint?: (name: string) => Promise<void>
  currentCheckpoint?: string
  availableModels: string[]
  currentModel: string
  onSelectModel: (model: string) => void
  modelInfoMap?: Record<string, { cached?: boolean; size_gb?: number }>
  souls: { name: string; traits?: string[]; description?: string }[]
  currentSoulName?: string
  onSwitchSoul?: (name: string) => void
}

export function CheckpointsTab({
  checkpoints,
  onLoadCheckpoint,
  currentCheckpoint,
  availableModels,
  currentModel,
  onSelectModel,
  modelInfoMap,
  souls,
  currentSoulName,
  onSwitchSoul,
}: CheckpointsTabProps) {
  return (
    <div className="space-y-2">
      {souls.length > 0 && (
        <div className="rounded-lg border border-border/40 bg-muted/10">
          <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30">Personality</div>
          <div className="max-h-28 overflow-y-auto p-1.5 space-y-0.5">
            {souls.map(s => (
              <button
                key={s.name}
                onClick={() => onSwitchSoul?.(s.name)}
                className={cn(
                  'w-full text-left px-2 py-1 rounded text-xs transition-colors flex items-center justify-between',
                  currentSoulName === s.name ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
                )}
              >
                <span>{s.name}</span>
                {currentSoulName === s.name && <IconCheck className="h-3 w-3 shrink-0" />}
              </button>
            ))}
          </div>
        </div>
      )}

      <ModelDropdown
        variant="panel"
      />

      <div className="rounded-lg border border-border/40 bg-muted/10">
        <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30 flex items-center justify-between">
          <span>Checkpoints</span>
          {checkpoints.length > 0 && <span className="text-[9px] font-normal normal-case">{checkpoints.length} saved</span>}
        </div>
        <div className="max-h-36 overflow-y-auto p-1.5 space-y-0.5">
          {checkpoints.length === 0 ? (
            <div className="px-2 py-3 text-[10px] text-muted-foreground text-center">No checkpoints yet</div>
          ) : checkpoints.map(ckpt => (
            <button
              key={ckpt.name}
              onClick={() => onLoadCheckpoint?.(ckpt.name)}
              className={cn(
                'w-full text-left px-2 py-1.5 rounded text-xs transition-colors',
                ckpt.is_loaded ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1 min-w-0">
                  <span className="truncate">{ckpt.name}</span>
                  {ckpt.is_loaded && <IconCheck className="h-3 w-3 shrink-0" />}
                </div>
                {ckpt.eval_verdict && (
                  <span className={cn(
                    'text-[9px] px-1 py-0 rounded shrink-0 ml-1',
                    ckpt.eval_verdict === 'PASS' ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning',
                  )}>
                    {ckpt.eval_verdict}
                  </span>
                )}
              </div>
              {(ckpt.loss != null || (ckpt.traits && ckpt.traits.length > 0)) && (
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
                  {ckpt.loss != null && <span>loss {ckpt.loss.toFixed(4)}</span>}
                  {ckpt.traits && ckpt.traits.length > 0 && (
                    <span className="truncate">{ckpt.traits.join(' · ')}</span>
                  )}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
