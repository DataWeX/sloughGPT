'use client'

/** Compare dialog: active soul vs a chosen soul, trait by trait. */
import { Dialog, DialogContent, DialogHeader, DialogTitle, cn } from '@sloughgpt/strui'
import { traitLabel } from './souls-format'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function SoulCompareDialog({ wb }: { wb: SoulsWorkbench }) {
  const { compareSoul, currentSoul, setCompareSoul, souls } = wb

  return (
    <Dialog
      open={compareSoul !== null}
      onOpenChange={(open) => {
        if (!open) setCompareSoul(null)
      }}
    >
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            Compare: {currentSoul} vs {compareSoul?.name}
          </DialogTitle>
        </DialogHeader>
        {compareSoul &&
          (() => {
            const activeSoul = souls.find((s) => s.name === currentSoul)
            if (!activeSoul) return null
            const allTraits = new Set<string>()
            Object.keys(activeSoul.personality || {}).forEach((t) => allTraits.add(t))
            Object.keys(compareSoul.personality || {}).forEach((t) => allTraits.add(t))
            const traits = Array.from(allTraits)
            return (
              <div className="space-y-3">
                {traits.map((trait) => {
                  const v1 = activeSoul.personality?.[trait] ?? 0
                  const v2 = compareSoul.personality?.[trait] ?? 0
                  const diff = v2 - v1
                  return (
                    <div key={trait} className="flex items-center gap-3 text-xs">
                      <span className="w-28 text-muted-foreground truncate">
                        {traitLabel(trait)}
                      </span>
                      <div className="flex-1 flex items-center gap-2">
                        <span className="w-10 text-right font-mono">{(v1 * 100).toFixed(0)}%</span>
                        <div className="flex-1 h-2 rounded-full bg-muted/50 overflow-hidden relative">
                          <div
                            className="absolute inset-0 h-full rounded-full bg-primary/40"
                            style={{ width: `${v1 * 100}%` }}
                          />
                          <div
                            className="absolute inset-0 h-full rounded-full bg-accent/60"
                            style={{ width: `${v2 * 100}%` }}
                          />
                        </div>
                        <span className="w-10 font-mono">{(v2 * 100).toFixed(0)}%</span>
                      </div>
                      <span
                        className={cn(
                          'w-12 text-right font-mono',
                          diff > 0
                            ? 'text-success'
                            : diff < 0
                              ? 'text-destructive'
                              : 'text-muted-foreground',
                        )}
                      >
                        {diff > 0 ? '+' : ''}
                        {(diff * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
                <div className="flex gap-4 text-xs text-muted-foreground pt-2 border-t border-border/30">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-primary/40" /> {activeSoul.name}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-accent/60" /> {compareSoul.name}
                  </span>
                </div>
              </div>
            )
          })()}
      </DialogContent>
    </Dialog>
  )
}
