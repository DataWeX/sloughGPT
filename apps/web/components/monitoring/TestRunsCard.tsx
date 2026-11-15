'use client'

import { memo, useCallback, useEffect, useState } from 'react'
import { Card, CardContent, FoldSection, Spinner } from '@sloughgpt/strui'
import { apiGet } from '@/lib/http-client'
import { timeAgo } from '@/lib/time-ago'

/** One raw entry as stored by the run suites (`{test, passed, detail}` rows). */
export interface TestRunEntry {
  test: string
  passed: boolean
  detail?: string
  /** Some suites (devtools telemetry) carry extra fields — kept as-is. */
  [key: string]: unknown
}

/** Per-file summary row served by the `test_results.list` capability. */
export interface TestRunFile {
  file: string
  status: 'passed' | 'failed'
  total: number
  passed: number
  failed: number
  mtime: number
  entries: TestRunEntry[]
}

interface TestRunsResponse {
  data?: { files?: TestRunFile[] }
}

/**
 * Test-run summaries from the `tests/core-py/test_results/*.json` store —
 * card 20261111_110106: these artifacts had zero consumers before this card.
 *
 * Summarize-don't-dump: per-file counts/status/freshness are the visible
 * surface; the raw entry arrays sit one click away behind `FoldSection`
 * (native `<details>` — collapsed by default). A failing run is the one thing
 * that must read at a glance, so it carries the destructive badge.
 */
export const TestRunsCard = memo(function TestRunsCard() {
  const [files, setFiles] = useState<TestRunFile[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  const fetchRuns = useCallback(async () => {
    setLoading(true)
    setError(false)
    try {
      const res = await apiGet<TestRunsResponse>('/test-results')
      setFiles(res.data?.files ?? [])
    } catch {
      setError(true)
      setFiles(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchRuns()
  }, [fetchRuns])

  return (
    <Card className="p-2.5">
      <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 block">
        Test runs
      </span>
      <CardContent className="p-0 space-y-1.5">
        {loading && (
          <p className="text-[10px] text-muted-foreground/60 text-center py-3" role="status">
            <Spinner />
          </p>
        )}

        {!loading && error && (
          <p className="text-[10px] text-destructive text-center py-3" role="alert">
            Could not load test runs
          </p>
        )}

        {!loading && !error && files?.length === 0 && (
          <p className="text-[10px] text-muted-foreground/60 text-center py-3">
            No test runs recorded yet
          </p>
        )}

        {!loading &&
          !error &&
          files?.map((f) => (
            <FoldSection
              key={f.file}
              heading={
                <span className="flex items-center justify-between gap-2 w-full min-w-0">
                  <span className="text-[10px] font-mono truncate min-w-0" title={f.file}>
                    {f.file}
                  </span>
                  <span className="flex shrink-0 items-center gap-1.5">
                    <span
                      className={`text-[8px] px-1 py-0.5 rounded font-medium tabular-nums ${
                        f.status === 'failed'
                          ? 'bg-destructive/10 text-destructive'
                          : 'bg-muted text-muted-foreground'
                      }`}
                    >
                      {f.status}
                    </span>
                    <span className="text-[9px] text-muted-foreground tabular-nums">
                      {f.passed}/{f.total}
                    </span>
                    <span className="text-[8px] text-muted-foreground/60 tabular-nums">
                      {timeAgo(f.mtime)}
                    </span>
                  </span>
                </span>
              }
            >
              <ul className="space-y-1" aria-label={`Raw entries for ${f.file}`}>
                {f.entries.map((e, i) => (
                  <li key={`${e.test}-${i}`} className="flex items-start gap-1.5 text-[9px]">
                    <span
                      className={`shrink-0 font-medium ${
                        e.passed ? 'text-muted-foreground' : 'text-destructive'
                      }`}
                      aria-label={e.passed ? 'passed' : 'failed'}
                    >
                      {e.passed ? '✓' : '✗'}
                    </span>
                    <span className="font-mono min-w-0 break-all">
                      {e.test}
                      {e.detail ? (
                        <>
                          <span aria-hidden="true" className="text-muted-foreground/40">
                            {' — '}
                          </span>
                          <span className="text-muted-foreground/60">{e.detail}</span>
                        </>
                      ) : null}
                    </span>
                  </li>
                ))}
              </ul>
            </FoldSection>
          ))}
      </CardContent>
    </Card>
  )
})
