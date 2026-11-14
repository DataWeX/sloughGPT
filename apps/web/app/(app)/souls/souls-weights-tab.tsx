'use client'

/** Weights tab: mode cards and the trait-weight slider editor with save/dirty state. */
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Slider,
  TabsContent,
} from '@sloughgpt/strui'
import { traitLabel } from './souls-format'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function WeightsTab({ wb }: { wb: SoulsWorkbench }) {
  const {
    editedWeights,
    handleSaveWeights,
    handleWeightChange,
    modes,
    savingWeights,
    setEditedWeights,
    traitWeights,
    weightsDirty,
  } = wb

  return (
    <TabsContent value="weights" className="space-y-4">
      {modes && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Context Modes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(modes).map(([key, mode]) => (
                <div key={key} className="rounded-md border border-border/40 p-3 space-y-1">
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    {key}
                  </div>
                  <div className="text-sm font-medium">{mode.label}</div>
                  <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all"
                      style={{ width: `${Math.max(0, Math.min(100, mode.confidence * 100))}%` }}
                    />
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {(mode.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Trait Weights</CardTitle>
            {weightsDirty && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-warning/10 text-warning font-medium">
                unsaved
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {weightsDirty && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  setEditedWeights(traitWeights ? JSON.parse(JSON.stringify(traitWeights)) : null)
                }
              >
                Reset
              </Button>
            )}
            <Button size="sm" onClick={handleSaveWeights} disabled={!weightsDirty || savingWeights}>
              {savingWeights ? 'Saving...' : 'Save'}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {editedWeights ? (
            <div className="space-y-6">
              {Object.entries(editedWeights).map(([category, weights]) => (
                <div key={category}>
                  <div className="text-xs font-medium text-muted-foreground mb-3 uppercase tracking-wider">
                    {category}
                  </div>
                  <div className="space-y-3">
                    {Object.entries(weights).map(([trait, value]) => (
                      <Slider
                        key={trait}
                        label={traitLabel(trait)}
                        showValue
                        value={[value]}
                        min={0}
                        max={1}
                        step={0.01}
                        formatValue={(v) => v.toFixed(2)}
                        onValueChange={([v]) => handleWeightChange(category, trait, v)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Loading trait weights...</p>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  )
}
