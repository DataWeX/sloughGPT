'use client'

import { cn, Card, CardHeader, CardTitle, CardContent, Button, SearchInput } from '@sloughgpt/strui'
import type { Soul } from '@/lib/souls-controller'
import { traitLabel, traitColor, sourceDir } from './soul-helpers'
import { formatShortDate } from '@/lib/time-format'

interface SoulListProps {
  souls: Soul[]
  currentSoul: string | null
  searchQuery: string
  onSearchChange: (query: string) => void
  switching: string | null
  onSwitch: (name: string) => void
  onSelectSoul: (soul: Soul) => void
}

function TraitRadar({ values, size = 80 }: { values: Record<string, number>; size?: number }) {
  const entries = Object.entries(values).slice(0, 8)
  if (entries.length === 0) return null
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 4
  const angleStep = (2 * Math.PI) / entries.length

  const points = entries.map((_, i) => {
    const angle = i * angleStep - Math.PI / 2
    const dist = r * Math.max(0.1, Math.min(1, entries[i][1]))
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) }
  })

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'

  return (
    <svg width={size} height={size} className="shrink-0" role="img" aria-label="Soul trait radar chart">
      {[0.25, 0.5, 0.75, 1].map(scale => (
        <polygon
          key={scale}
          points={entries.map((_, i) => {
            const angle = i * angleStep - Math.PI / 2
            return `${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`
          }).join(' ')}
          fill="none"
          stroke="currentColor"
          className="text-border/40"
          strokeWidth={0.5}
        />
      ))}
      <polygon points={pathD} fill="rgb(var(--primary))" fillOpacity={0.15} stroke="rgb(var(--primary))" strokeWidth={1} />
    </svg>
  )
}

export function SoulList({ souls, currentSoul, searchQuery, onSearchChange, switching, onSwitch, onSelectSoul }: SoulListProps) {
  const filtered = souls.filter(s =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.traits?.some(t => t.toLowerCase().includes(searchQuery.toLowerCase())) ||
    s.lineage?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Personalities</CardTitle>
        <SearchInput
          value={searchQuery}
          onChange={onSearchChange}
          placeholder="Search souls..."
          className="max-w-xs"
        />
      </CardHeader>
      <CardContent>
        {filtered.length === 0 ? (
          <div className="text-center py-8 space-y-2">
            <p className="text-sm text-muted-foreground">
              {searchQuery ? 'No personalities match your search.' : 'No personalities found.'}
            </p>
            {!searchQuery && (
              <p className="text-xs text-muted-foreground">
                Souls are loaded from <code className="bg-muted px-1 rounded">.soul</code> files in <code className="bg-muted px-1 rounded">models/</code>.
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map(soul => (
              <div
                key={soul.name}
                className={cn('flex items-center justify-between rounded-md border px-3 py-2.5 text-sm transition-colors cursor-pointer group', currentSoul === soul.name ? 'border-primary/40 bg-primary/[0.08]' : 'border-border/60 hover:bg-muted/50')}
                onClick={() => onSelectSoul(soul)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectSoul(soul); } }}
                role="button"
                tabIndex={0}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <TraitRadar values={soul.personality || {}} size={48} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">{soul.name}</span>
                      {currentSoul === soul.name && (
                        <span className="text-xs font-medium bg-primary/10 text-primary px-1.5 py-0.5 rounded-full">active</span>
                      )}
                      {soul.version && (
                        <span className="text-xs font-mono text-muted-foreground">v{soul.version}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                      {soul.lineage && <span>{soul.lineage}</span>}
                      {soul.size_mb != null && soul.size_mb > 0 && <span>{soul.size_mb.toFixed(1)} MB</span>}
                      {soul.born_at && <span>{formatShortDate(soul.born_at)}</span>}
                      {soul.epochs_trained != null && soul.epochs_trained > 0 && <span>{soul.epochs_trained} epochs</span>}
                      {soul.final_val_loss != null && <span>val {soul.final_val_loss.toFixed(3)}</span>}
                      {soul.training_dataset && (
                        <span className="font-mono text-xs">{sourceDir(soul.training_dataset)}</span>
                      )}
                    </div>
                    {soul.traits && soul.traits.length > 0 && (
                      <div className="flex gap-1 mt-1.5 flex-wrap">
                        {soul.traits.slice(0, 5).map(trait => (
                          <span key={trait} className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{trait}</span>
                        ))}
                        {soul.traits.length > 5 && (
                          <span className="text-xs text-muted-foreground">+{soul.traits.length - 5}</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {soul.personality && Object.keys(soul.personality).length > 0 && (
                    <span className={cn('text-xs font-mono', traitColor(Object.values(soul.personality).reduce((a, b) => a + b, 0) / Object.values(soul.personality).length))}>
                      {(Object.values(soul.personality).reduce((a, b) => a + b, 0) / Object.values(soul.personality).length * 100).toFixed(0)}%
                    </span>
                  )}
                  {currentSoul !== soul.name && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => { e.stopPropagation(); onSwitch(soul.name) }}
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
  )
}
