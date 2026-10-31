'use client'

/**
 * Dashboard header halves: the headline card rendered into `PageContainer`'s
 * `title` slot and the action cluster rendered into its `headerRight` slot.
 * Both are pure — live state, timestamps and handlers arrive as props from
 * `useConsciousnessDashboard`.
 */
import { Button, Card, CardContent, StatusDot, Switch } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { formatTimeAgo } from './dashboard-format'

interface DashboardHeadlineProps {
  isLive: boolean
  lastUpdate: number | null
  onToggleLive: () => void
}

export function DashboardHeadline({ isLive, lastUpdate, onToggleLive }: DashboardHeadlineProps) {
  const { t } = useLocale()

  return (
    <Card className="border-primary/20 bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]">
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-2">
              <StatusDot
                tone="success"
                pulse
                showLabel
                label={
                  isLive ? t('consciousness_dashboard.live') : t('consciousness_dashboard.autoRefresh')
                }
              />
            </div>
            <h1 className="sl-h1">Consciousness Dashboard</h1>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Switch checked={isLive} onCheckedChange={onToggleLive} aria-label="Toggle live updates" />
          </div>
        </div>
        {lastUpdate && (
          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/40">
            <span className="text-xs text-muted-foreground">
              Updated {formatTimeAgo(lastUpdate / 1000)}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

interface DashboardActionsProps {
  seeding: boolean
  reflecting: boolean
  exporting: boolean
  importing: boolean
  onSeed: () => void
  onReflect: () => void
  onExport: () => void
  onImport: () => void
}

export function DashboardActions({
  seeding,
  reflecting,
  exporting,
  importing,
  onSeed,
  onReflect,
  onExport,
  onImport,
}: DashboardActionsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      <Button size="sm" variant="outline" onClick={onSeed} disabled={seeding}>
        {seeding ? 'Seeding...' : 'Seed 30'}
      </Button>
      <Button size="sm" variant="outline" onClick={onReflect} disabled={reflecting}>
        {reflecting ? 'Reflecting...' : 'Reflect'}
      </Button>
      <Button size="sm" variant="outline" onClick={onExport} disabled={exporting}>
        {exporting ? 'Exporting...' : 'Export'}
      </Button>
      <Button size="sm" variant="outline" onClick={onImport} disabled={importing}>
        {importing ? 'Importing...' : 'Import'}
      </Button>
    </div>
  )
}
