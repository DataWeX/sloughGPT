'use client'

import { cn, Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, Button } from '@sloughgpt/strui'
import type { Soul } from '@/lib/souls-controller'
import { traitLabel, traitColor } from './soul-helpers'
import { formatShortDate } from '@/lib/time-format'

function PersonalityBar({ label, value, color = 'bg-primary' }: { label: string; value: number; color?: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-0.5">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn('font-mono font-medium', traitColor(value))}>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
        <div className={cn('h-full rounded-full transition-all', color)} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  )
}

interface SoulDetailDialogProps {
  soul: Soul | null
  currentSoul: string | null
  onClose: () => void
  onSwitch: (name: string) => void
}

export function SoulDetailDialog({ soul, currentSoul, onClose, onSwitch }: SoulDetailDialogProps) {
  return (
    <Dialog open={soul !== null} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {soul?.name}
            {soul?.version && <span className="text-xs font-mono text-muted-foreground">v{soul.version}</span>}
          </DialogTitle>
          {soul?.description && (
            <DialogDescription>{soul.description}</DialogDescription>
          )}
        </DialogHeader>
        {soul && (
          <div className="space-y-4">
            {/* Training Metadata */}
            {(soul.born_at || soul.lineage || soul.training_dataset || (soul.epochs_trained != null && soul.epochs_trained > 0) || (soul.size_mb != null && soul.size_mb > 0)) && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Training Info</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {soul.lineage && <div><span className="text-muted-foreground">Lineage:</span> <span className="font-medium">{soul.lineage}</span></div>}
                  {soul.base_model && <div><span className="text-muted-foreground">Base:</span> <span className="font-medium">{soul.base_model}</span></div>}
                  {soul.born_at && <div><span className="text-muted-foreground">Created:</span> <span className="font-medium">{formatShortDate(soul.born_at)}</span></div>}
                  {(soul.size_mb != null && soul.size_mb > 0) && <div><span className="text-muted-foreground">Size:</span> <span className="font-medium">{soul.size_mb.toFixed(1)} MB</span></div>}
                  {(soul.epochs_trained != null && soul.epochs_trained > 0) && <div><span className="text-muted-foreground">Epochs:</span> <span className="font-medium">{soul.epochs_trained}</span></div>}
                  {soul.final_train_loss != null && <div><span className="text-muted-foreground">Train loss:</span> <span className="font-medium">{soul.final_train_loss.toFixed(4)}</span></div>}
                  {soul.final_val_loss != null && <div><span className="text-muted-foreground">Val loss:</span> <span className="font-medium">{soul.final_val_loss.toFixed(4)}</span></div>}
                  {soul.training_dataset && <div className="col-span-2"><span className="text-muted-foreground">Dataset:</span> <span className="font-medium font-mono text-xs">{soul.training_dataset.split('/').pop()}</span></div>}
                </div>
              </div>
            )}

            {/* Personality */}
            {soul.personality && Object.keys(soul.personality).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Personality</div>
                <div className="space-y-2">
                  {Object.entries(soul.personality).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                    <PersonalityBar key={key} label={traitLabel(key)} value={value} />
                  ))}
                </div>
              </div>
            )}

            {/* Traits */}
            {soul.traits && soul.traits.length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Traits</div>
                <div className="flex flex-wrap gap-1.5">
                  {soul.traits.map(t => (
                    <span key={t} className="text-xs px-2 py-1 rounded bg-primary/10 text-primary font-medium">{t}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Behavior */}
            {soul.behavior && Object.keys(soul.behavior).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Behavior</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(soul.behavior).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{traitLabel(key)}: </span>
                      <span className="font-medium">
                        {typeof value === 'number' ? (value * 100).toFixed(0) + '%' : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cognition */}
            {soul.cognition && Object.keys(soul.cognition).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Cognition</div>
                <div className="space-y-2">
                  {Object.entries(soul.cognition).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                    <PersonalityBar key={key} label={traitLabel(key)} value={value} color="bg-accent" />
                  ))}
                </div>
              </div>
            )}

            {/* Emotion */}
            {soul.emotion && Object.keys(soul.emotion).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Emotion</div>
                <div className="space-y-2">
                  {Object.entries(soul.emotion).sort((a, b) => b[1] - a[1]).map(([key, value]) => (
                    <PersonalityBar key={key} label={traitLabel(key)} value={value} color="bg-success" />
                  ))}
                </div>
              </div>
            )}

            {/* Generation Params */}
            {soul.generation_params && Object.keys(soul.generation_params).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">Generation</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {Object.entries(soul.generation_params).map(([key, value]) => (
                    <div key={key}>
                      <span className="text-muted-foreground">{traitLabel(key)}: </span>
                      <span className="font-mono font-medium">{typeof value === 'number' ? value : String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Source Path */}
            {soul.path && (
              <div className="text-xs text-muted-foreground font-mono truncate pt-1 border-t border-border/30">
                {soul.path}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <Button size="sm" variant="outline" onClick={onClose}>Close</Button>
              {currentSoul !== soul.name && (
                <Button size="sm" onClick={() => { onSwitch(soul.name); onClose() }}>
                  Switch to {soul.name}
                </Button>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
