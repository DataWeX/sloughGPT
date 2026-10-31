'use client'

/**
 * Focal metrics + the Response Quality / Reflection pair.
 *
 * Derivation lives here rather than in the page: `score` and `avgGrowth` are
 * shaped from the raw episode/eval payloads this panel receives, so the page
 * only forwards data.
 */
import { Card, CardContent, CardDescription, CardHeader, CardTitle, cn, KpiGrid, SectionHeader, StatCard } from '@sloughgpt/strui'
import type { Episode, EvalReport } from '@/hooks/useConsciousnessDashboard'
import { levelName } from './dashboard-format'

interface DashboardMetricsProps {
  totalEpisodes: number
  episodes: Episode[]
  evalReport: EvalReport | null
  consciousnessLevel: number
  isLive: boolean
}

export function DashboardMetrics({
  totalEpisodes,
  episodes,
  evalReport,
  consciousnessLevel,
  isLive,
}: DashboardMetricsProps) {
  const score = evalReport ? evalReport.overall_score : null

  const avgGrowth =
    episodes.length > 0
      ? `${((episodes.reduce((s, e) => s + e.growth_delta, 0) / episodes.length) * 100).toFixed(1)}%`
      : '—'

  return (
    <>
      {/* Focal metrics */}
      <KpiGrid columns={4}>
        <StatCard label="Total Episodes" value={totalEpisodes} numeric />
        <StatCard
          label="Quality Score"
          value={score !== null ? `${score.toFixed(0)}/100` : '—'}
          numeric={score !== null}
          description="How well responses match personality goals"
        />
        <StatCard label="Avg Growth" value={avgGrowth} numeric />
        <StatCard
          label="Consciousness Level"
          value={levelName(consciousnessLevel)}
          description={`Depth of reflection · ${isLive ? 'live' : 'auto-refresh'}`}
        />
      </KpiGrid>

      {/* Response quality & reflection */}
      <SectionHeader
        title="Response Quality & Reflection"
        description="How responses track personality goals, and the latest self-insight"
      />
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Response Quality</CardTitle>
            <CardDescription>How well responses match personality goals</CardDescription>
          </CardHeader>
          <CardContent>
            {score !== null ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Overall Score</span>
                  <span className="text-xl font-bold tracking-tight">{score.toFixed(0)}/100</span>
                </div>
                <div
                  className="h-2 rounded-full bg-muted overflow-hidden"
                  role="progressbar"
                  aria-valuenow={score}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      score > 70 ? 'bg-success' : score > 40 ? 'bg-warning' : 'bg-destructive',
                    )}
                    style={{ width: `${score}%` }}
                  />
                </div>
                {evalReport?.diagnostics?.slice(0, 3).map((d: string, i: number) => (
                  <div key={i} className="text-xs text-muted-foreground">
                    {d}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">No evaluation data yet</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Last Reflection</CardTitle>
            <CardDescription>What the system is thinking right now</CardDescription>
          </CardHeader>
          <CardContent>
            {(() => {
              // `last_reflection` rides the report's index signature; the
              // explicit cast keeps it a renderable string instead of unknown.
              const lastReflection =
                (evalReport?.last_reflection as string | undefined) ||
                episodes[episodes.length - 1]?.self_insight
              return lastReflection ? (
                <div className="text-sm italic text-muted-foreground">
                  &quot;{lastReflection}&quot;
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">
                  No reflections yet. Click &quot;Reflect&quot; to generate one.
                </div>
              )
            })()}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
