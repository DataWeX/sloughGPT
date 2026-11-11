'use client'

import { memo } from 'react'
import {
  Button,
  Card,
  CardContent,
  FoldSection,
  KpiGrid,
  Skeleton,
  StatCard,
  StatusBadge,
} from '@sloughgpt/strui'
import { useEndpoint } from '@/lib/protocol/useEndpoint'
import { timeAgo } from '@/lib/time-ago'

export interface TestRunItem {
  test: string
  passed: boolean | null
  detail: string
}

export interface TestRunSuite {
  file: string
  mtime: string
  total: number
  passed: number
  failed: number
  unknown: number
  items: TestRunItem[]
}

export interface TestRunsData {
  suites: TestRunSuite[]
  total: number
  passed: number
  failed: number
  unknown: number
}

function outcomeOf(passed: boolean | null): { label: string; tone: 'success' | 'destructive' | 'warning' } {
  if (passed === true) return { label: 'Passed', tone: 'success' }
  if (passed === false) return { label: 'Failed', tone: 'destructive' }
  return { label: 'Unknown', tone: 'warning' }
}

const HEADING =
  'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block'

/**
 * Journey test-run evidence (tests/core-py/test_results/*.json) on the
 * monitoring surface — the files had zero consumers before this card.
 *
 * Summary first (KPIs + relative last-run time), raw per-test detail one
 * click away in a collapsed fold — a report, not a dump (SOP 15).
 * Loading is the explicit status boolean, never inferred from emptiness.
 */
export const TestRunsCard = memo(function TestRunsCard() {
  const { data, status, error, refetch } = useEndpoint<TestRunsData>('test_results.list')
  const loading = status === 'idle' || status === 'loading'

  if (loading) {
    return (
      <Card className="p-3">
        <span className={HEADING}>Test Runs</span>
        <CardContent className="p-0">
          <div className="animate-pulse space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-2/3" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="p-3">
        <span className={HEADING}>Test Runs</span>
        <CardContent className="p-0">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm text-destructive">{error}</p>
            <Button variant="outline" size="sm" onClick={refetch}>
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const suites = data?.suites ?? []
  if (suites.length === 0) {
    return (
      <Card className="p-3">
        <span className={HEADING}>Test Runs</span>
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">No test runs recorded yet</p>
        </CardContent>
      </Card>
    )
  }

  const newest = suites[0]?.mtime
  const failed = data?.failed ?? 0

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          Test Runs
        </span>
        {newest && (
          <span className="text-[10px] text-muted-foreground">Last run {timeAgo(newest)}</span>
        )}
      </div>
      <CardContent className="p-0">
        <KpiGrid columns={4}>
          <StatCard label="Passed" value={data?.passed ?? 0} numeric />
          <StatCard label="Failed" value={failed} numeric />
          <StatCard label="Unknown" value={data?.unknown ?? 0} numeric />
          <StatCard label="Total" value={data?.total ?? 0} numeric />
        </KpiGrid>
        {failed > 0 && (
          <p className="mt-2 text-xs text-destructive">
            {failed} failing — raw runs below
          </p>
        )}
        <FoldSection
          heading={`Raw runs (${suites.length} suite${suites.length === 1 ? '' : 's'})`}
          className="mt-3"
          open={false}
        >
          <div className="space-y-3">
            {suites.map((suite) => (
              <div key={suite.file}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-medium text-foreground">{suite.file}</span>
                  <span className="text-[10px] text-muted-foreground">
                    {suite.passed}/{suite.total} passed · {timeAgo(suite.mtime)}
                  </span>
                </div>
                <ul className="mt-1 space-y-1">
                  {suite.items.map((item) => {
                    const outcome = outcomeOf(item.passed)
                    return (
                      <li key={item.test} className="flex items-start justify-between gap-2 text-xs">
                        <span className="min-w-0 text-muted-foreground">
                          {item.test}
                          {item.passed !== true && item.detail && (
                            <span className="block text-[10px] text-destructive/80">
                              {item.detail}
                            </span>
                          )}
                        </span>
                        <StatusBadge tone={outcome.tone} size="sm">
                          {outcome.label}
                        </StatusBadge>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </div>
        </FoldSection>
      </CardContent>
    </Card>
  )
})
