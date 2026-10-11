/**
 * Response quality & reflection — the evaluation score bar with diagnostics,
 * alongside the most recent self-insight.
 */

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SectionHeader,
  cn,
} from '@sloughgpt/strui'
import type { Episode, EvalReport } from './types'

export interface DashboardResponseQualityProps {
  score: number | null
  evalReport: EvalReport | null
  episodes: Episode[]
}

export function DashboardResponseQuality({
  score,
  evalReport,
  episodes,
}: DashboardResponseQualityProps) {
  const lastReflection =
    (evalReport as { last_reflection?: string } | null)?.last_reflection ||
    episodes[episodes.length - 1]?.self_insight

  return (
    <>
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
              <div className="text-sm italic text-muted-foreground">"{lastReflection}"</div>
            ) : (
              <div className="text-sm text-muted-foreground">
                No reflections yet. Click "Reflect" to generate one.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
