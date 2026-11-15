'use client'

// Focal metrics + response quality/reflection cards. Extracted from page.tsx
// by the page-decomposition refactor — pure presentation over hook outputs.

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  KpiGrid,
  SectionHeader,
  StatCard,
  cn,
} from '@sloughgpt/strui'
import { levelName, type EvalReport } from '../dashboard-data'

interface DashboardSummaryProps {
  totalEpisodes: number
  score: number | null
  avgGrowth: string
  consciousnessLevel: number
  isLive: boolean
  evalReport: EvalReport | null
  lastReflection: string | null
}

export function DashboardSummary({
  totalEpisodes,
  score,
  avgGrowth,
  consciousnessLevel,
  isLive,
  evalReport,
  lastReflection,
}: DashboardSummaryProps) {
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
            {lastReflection ? (
              <div className="text-sm italic text-muted-foreground">&quot;{lastReflection}&quot;</div>
            ) : (
              <div className="text-sm text-muted-foreground">
                No reflections yet. Click &quot;Reflect&quot; to generate one.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
