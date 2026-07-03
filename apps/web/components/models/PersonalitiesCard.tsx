'use client'

import { Badge, Chip } from '@/components/ui/tags'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select'
import { cn } from '@/lib/cn'
import type { Soul, Checkpoint } from '@/lib/souls-controller'

interface PersonalitiesCardProps {
  souls: Soul[]
  soulsLoading: boolean
  checkpoints: Checkpoint[]
  checkpointsLoading: boolean
  currentSoul: string | null
  activeCheckpoint: string | null
  switchingSoul: string | null
  onSwitch: (name: string, checkpointName?: string) => void
}

export default function PersonalitiesCard({
  souls, soulsLoading, checkpoints, checkpointsLoading,
  currentSoul, activeCheckpoint, switchingSoul, onSwitch,
}: PersonalitiesCardProps) {
  if (souls.length === 0 && !soulsLoading) return null

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Personalities</CardTitle></CardHeader>
      <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {soulsLoading ? (
          [1,2,3].map(i => (
            <div key={i} className="animate-pulse flex items-center gap-3 p-3 rounded-lg border border-border/60">
              <div className="w-2 h-2 rounded-full bg-muted-foreground/20" />
              <div className="flex-1 space-y-1.5">
                <div className="h-3 w-24 bg-muted rounded" />
                <div className="h-2.5 w-32 bg-muted rounded" />
              </div>
              <div className="h-6 w-14 bg-muted rounded" />
            </div>
          ))
        ) : (
          souls.map((s: Soul) => {
            const soulCheckpoints = checkpoints?.filter((c: Checkpoint) => c.soul === s.name) ?? []
            const isCurrent = currentSoul === s.name
            return (
              <div key={s.name} className={cn("flex items-center justify-between p-3 rounded-lg border transition-all duration-200 hover:-translate-y-0.5 hover:shadow-sm", isCurrent ? "border-primary/40 bg-primary/5" : "border-border/60")}>
                <div className="flex items-center gap-3 min-w-0">
                  <div className={cn("w-2 h-2 rounded-full shrink-0", isCurrent ? "bg-primary" : "bg-muted-foreground/30")} />
                  <div className="min-w-0">
                    <div className="text-sm font-medium truncate">{s.name}</div>
                    <div className="text-xs text-muted-foreground truncate">{s.description || ''}</div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  {soulCheckpoints.length > 0 ? (
                    <Select
                      value={isCurrent && activeCheckpoint ? activeCheckpoint : ''}
                      onValueChange={(val) => {
                        if (val === '__base__') onSwitch(s.name)
                        else if (val) onSwitch(s.name, val)
                      }}
                      disabled={switchingSoul === s.name}
                    >
                      <SelectTrigger className="h-7 text-xs w-auto min-w-[3.5rem] px-2">
                        <SelectValue placeholder="Switch" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__base__">Switch</SelectItem>
                        {soulCheckpoints.map((cp: Checkpoint) => (
                          <SelectItem key={cp.name} value={cp.name}>
                            {cp.name}{activeCheckpoint === cp.name ? ' (active)' : ''}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : checkpointsLoading ? (
                    <span className="text-xs text-muted-foreground animate-pulse">loading&hellip;</span>
                  ) : isCurrent ? (
                    <Badge label="Active" variant="success" />
                  ) : (
                    <Button variant="outline" size="sm" className="h-7 text-xs px-2" disabled={switchingSoul === s.name} onClick={() => onSwitch(s.name)}>
                      {switchingSoul === s.name ? '...' : 'Switch'}
                    </Button>
                  )}
                </div>
              </div>
            )
          })
        )}
      </CardContent>
    </Card>
  )
}
