/**
 * Dashboard header card — live/auto-refresh status dot, page title, and the
 * last-update stamp. Rendered into `PageContainer`'s `title` slot.
 */

import { Card, CardContent, StatusDot, Switch } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { formatTimeAgo } from './types'

export interface DashboardHeaderProps {
  isLive: boolean
  lastUpdate: number | null
  onToggleLive: (checked: boolean) => void
}

export function DashboardHeader({ isLive, lastUpdate, onToggleLive }: DashboardHeaderProps) {
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
                  isLive
                    ? t('consciousness_dashboard.live')
                    : t('consciousness_dashboard.autoRefresh')
                }
              />
            </div>
            <h1 className="sl-h1">Consciousness Dashboard</h1>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Switch
              checked={isLive}
              onCheckedChange={onToggleLive}
              aria-label="Toggle live updates"
            />
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
