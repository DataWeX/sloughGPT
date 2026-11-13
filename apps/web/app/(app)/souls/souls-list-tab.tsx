'use client'

/** Souls tab: search, personality cards, switch-to-soul and detail/compare triggers. */
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  SearchInput,
  TabsContent,
  cn,
} from '@sloughgpt/strui'
import { SoulPersonalityCard } from '@/components/souls/SoulPersonalityCard'
import { formatShortDate } from '@/lib/time-format'
import { sourceDir, traitColor } from './souls-format'
import { TraitRadar } from './souls-panels'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function SoulsListTab({ wb }: { wb: SoulsWorkbench }) {
  const {
    currentSoul,
    filteredSouls,
    handleSwitch,
    searchQuery,
    setDetailSoul,
    setSearchQuery,
    souls,
    switching,
  } = wb

  return (
    <TabsContent value="souls" className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Personalities</CardTitle>
          <SearchInput
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search souls..."
            className="max-w-xs"
          />
        </CardHeader>
        <CardContent>
          {filteredSouls.length === 0 ? (
            <div className="text-center py-8 space-y-2">
              <p className="text-sm text-muted-foreground">
                {searchQuery ? 'No personalities match your search.' : 'No personalities found.'}
              </p>
              {!searchQuery && (
                <p className="text-xs text-muted-foreground">
                  Souls are loaded from <code className="bg-muted px-1 rounded">.soul</code> files
                  in <code className="bg-muted px-1 rounded">models/</code>.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredSouls.map((soul) => (
                <div
                  key={soul.name}
                  className={cn(
                    'flex items-center justify-between rounded-md border px-3 py-2.5 text-sm transition-colors cursor-pointer group',
                    currentSoul === soul.name
                      ? 'border-primary/40 bg-primary/[0.08]'
                      : 'border-border/40 hover:bg-muted/50',
                  )}
                  onClick={() => setDetailSoul(soul)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setDetailSoul(soul)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <TraitRadar values={soul.personality || {}} size={48} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{soul.name}</span>
                        {currentSoul === soul.name && (
                          <span className="text-xs font-medium bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">
                            active
                          </span>
                        )}
                        {soul.version && (
                          <span className="text-xs font-mono text-muted-foreground">
                            v{soul.version}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                        {soul.lineage && <span>{soul.lineage}</span>}
                        {soul.size_mb != null && soul.size_mb > 0 && (
                          <span>{soul.size_mb.toFixed(1)} MB</span>
                        )}
                        {soul.born_at && <span>{formatShortDate(soul.born_at)}</span>}
                        {soul.epochs_trained != null && soul.epochs_trained > 0 && (
                          <span>{soul.epochs_trained} epochs</span>
                        )}
                        {soul.final_val_loss != null && (
                          <span>val {soul.final_val_loss.toFixed(3)}</span>
                        )}
                        {soul.training_dataset && (
                          <span className="font-mono text-xs">
                            {sourceDir(soul.training_dataset)}
                          </span>
                        )}
                      </div>
                      {soul.traits && soul.traits.length > 0 && (
                        <div className="flex gap-1 mt-1.5 flex-wrap">
                          {soul.traits.slice(0, 5).map((trait) => (
                            <span
                              key={trait}
                              className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium"
                            >
                              {trait}
                            </span>
                          ))}
                          {soul.traits.length > 5 && (
                            <span className="text-xs text-muted-foreground">
                              +{soul.traits.length - 5}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {soul.personality && Object.keys(soul.personality).length > 0 && (
                      <span
                        className={cn(
                          'text-xs font-mono',
                          traitColor(
                            Object.values(soul.personality).reduce((a, b) => a + b, 0) /
                              Object.values(soul.personality).length,
                          ),
                        )}
                      >
                        {(
                          (Object.values(soul.personality).reduce((a, b) => a + b, 0) /
                            Object.values(soul.personality).length) *
                          100
                        ).toFixed(0)}
                        %
                      </span>
                    )}
                    {currentSoul !== soul.name && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleSwitch(soul.name)
                        }}
                        disabled={switching === soul.name}
                      >
                        {switching === soul.name ? 'Switching...' : 'Switch'}
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {(() => {
        const activeSoul = souls.find((s) => s.name === currentSoul)
        if (!activeSoul?.personality || Object.keys(activeSoul.personality).length === 0)
          return null
        return (
          <SoulPersonalityCard
            personality={activeSoul.personality}
            traits={activeSoul.traits}
            soulName={activeSoul.name}
          />
        )
      })()}
    </TabsContent>
  )
}
