'use client'

import type { ReactNode } from 'react'
import { cn } from '@sloughgpt/strui'

export type StatTile = {
  key: string
  label: string
  value: ReactNode
  hint?: ReactNode
}

/**
 * Live-training stat tiles.
 *
 * One data-driven row replaces the six hand-rolled conditional tiles that were
 * copy-pasted inside both QuickTrainCard and TrainingPipeline (and inside each
 * of them six times). The layout is flex-wrap with a gap-compensated basis
 * (50% - 4px mobile, 25% - 6px from sm) plus grow: tiles that are absent
 * (loss/eta/speed are conditional) never leave an empty cell the way the old
 * grid-cols-4 did at 6 items, and the trailing row stretches to full width
 * instead of leaving a hole.
 */
export function StatTileGrid({ tiles, className }: { tiles: StatTile[]; className?: string }) {
  if (tiles.length === 0) return null
  return (
    <div className={cn('flex flex-wrap gap-2', className)}>
      {tiles.map((tile) => (
        <div
          key={tile.key}
          className="min-w-0 grow basis-[calc(50%_-_4px)] rounded-lg bg-muted/30 px-3 py-2 sm:basis-[calc(25%_-_6px)]"
        >
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60">
            {tile.label}
          </p>
          <p className="text-sm font-semibold tabular-nums">
            {tile.value}
            {tile.hint !== undefined && (
              <span className="text-xs font-normal text-muted-foreground/40">{tile.hint}</span>
            )}
          </p>
        </div>
      ))}
    </div>
  )
}
