'use client'

/** Soul detail dialog: full metadata, personality bars and switch/compare actions. */
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@sloughgpt/strui'
import { formatShortDate } from '@/lib/time-format'
import { traitLabel } from './souls-format'
import { PersonalityBar } from './souls-panels'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function SoulDetailDialog({ wb }: { wb: SoulsWorkbench }) {
  const { currentSoul, detailSoul, handleSwitch, setDetailSoul } = wb

  return (
    <Dialog
      open={detailSoul !== null}
      onOpenChange={(open) => {
        if (!open) setDetailSoul(null)
      }}
    >
      <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {detailSoul?.name}
            {detailSoul?.version && (
              <span className="text-xs font-mono text-muted-foreground">v{detailSoul.version}</span>
            )}
          </DialogTitle>
          {detailSoul?.description && (
            <DialogDescription>{detailSoul.description}</DialogDescription>
          )}
        </DialogHeader>
        {detailSoul && (
          <div className="space-y-4">
            {/* Training Metadata */}
            {(detailSoul.born_at ||
              detailSoul.lineage ||
              detailSoul.training_dataset ||
              (detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0) ||
              (detailSoul.size_mb != null && detailSoul.size_mb > 0)) && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Training Info
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {detailSoul.lineage && (
                    <div>
                      <span className="text-muted-foreground">Lineage:</span>{' '}
                      <span className="font-medium">{detailSoul.lineage}</span>
                    </div>
                  )}
                  {detailSoul.base_model && (
                    <div>
                      <span className="text-muted-foreground">Base:</span>{' '}
                      <span className="font-medium">{detailSoul.base_model}</span>
                    </div>
                  )}
                  {detailSoul.born_at && (
                    <div>
                      <span className="text-muted-foreground">Created:</span>{' '}
                      <span className="font-medium">{formatShortDate(detailSoul.born_at)}</span>
                    </div>
                  )}
                  {detailSoul.size_mb != null && detailSoul.size_mb > 0 && (
                    <div>
                      <span className="text-muted-foreground">Size:</span>{' '}
                      <span className="font-medium">{detailSoul.size_mb.toFixed(1)} MB</span>
                    </div>
                  )}
                  {detailSoul.epochs_trained != null && detailSoul.epochs_trained > 0 && (
                    <div>
                      <span className="text-muted-foreground">Epochs:</span>{' '}
                      <span className="font-medium">{detailSoul.epochs_trained}</span>
                    </div>
                  )}
                  {detailSoul.final_train_loss != null && (
                    <div>
                      <span className="text-muted-foreground">Train loss:</span>{' '}
                      <span className="font-medium">{detailSoul.final_train_loss.toFixed(4)}</span>
                    </div>
                  )}
                  {detailSoul.final_val_loss != null && (
                    <div>
                      <span className="text-muted-foreground">Val loss:</span>{' '}
                      <span className="font-medium">{detailSoul.final_val_loss.toFixed(4)}</span>
                    </div>
                  )}
                  {detailSoul.training_dataset && (
                    <div className="col-span-2">
                      <span className="text-muted-foreground">Dataset:</span>{' '}
                      <span className="font-medium font-mono text-xs">
                        {detailSoul.training_dataset.split('/').pop()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Personality */}
            {detailSoul.personality && Object.keys(detailSoul.personality).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Personality
                </div>
                <div className="space-y-2">
                  {Object.entries(detailSoul.personality)
                    .sort((a, b) => b[1] - a[1])
                    .map(([key, value]) => (
                      <PersonalityBar key={key} label={traitLabel(key)} value={value} />
                    ))}
                </div>
              </div>
            )}

            {/* Traits */}
            {detailSoul.traits && detailSoul.traits.length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Traits
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {detailSoul.traits.map((t) => (
                    <span
                      key={t}
                      className="text-xs px-2 py-1 rounded bg-primary/10 text-primary font-medium"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Behavior */}
            {detailSoul.behavior && Object.keys(detailSoul.behavior).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Behavior
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(detailSoul.behavior).map(([key, value]) => (
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
            {detailSoul.cognition && Object.keys(detailSoul.cognition).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Cognition
                </div>
                <div className="space-y-2">
                  {Object.entries(detailSoul.cognition)
                    .sort((a, b) => b[1] - a[1])
                    .map(([key, value]) => (
                      <PersonalityBar
                        key={key}
                        label={traitLabel(key)}
                        value={value}
                        color="bg-accent"
                      />
                    ))}
                </div>
              </div>
            )}

            {/* Emotion */}
            {detailSoul.emotion && Object.keys(detailSoul.emotion).length > 0 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                  Emotion
                </div>
                <div className="space-y-2">
                  {Object.entries(detailSoul.emotion)
                    .sort((a, b) => b[1] - a[1])
                    .map(([key, value]) => (
                      <PersonalityBar
                        key={key}
                        label={traitLabel(key)}
                        value={value}
                        color="bg-success"
                      />
                    ))}
                </div>
              </div>
            )}

            {/* Generation Params */}
            {detailSoul.generation_params &&
              Object.keys(detailSoul.generation_params).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                    Generation
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    {Object.entries(detailSoul.generation_params).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-muted-foreground">{traitLabel(key)}: </span>
                        <span className="font-mono font-medium">
                          {typeof value === 'number' ? value : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* Source Path */}
            {detailSoul.path && (
              <div className="text-xs text-muted-foreground font-mono truncate pt-1 border-t border-border/30">
                {detailSoul.path}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <Button size="sm" variant="outline" onClick={() => setDetailSoul(null)}>
                Close
              </Button>
              {currentSoul !== detailSoul.name && (
                <Button
                  size="sm"
                  onClick={() => {
                    handleSwitch(detailSoul.name)
                    setDetailSoul(null)
                  }}
                >
                  Switch to {detailSoul.name}
                </Button>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
