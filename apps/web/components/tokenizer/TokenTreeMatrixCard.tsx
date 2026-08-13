'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Skeleton, Chip } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { tokenTreeController, type MatrixSummary } from '@/lib/token-tree-controller'

const TOP_K = 8

const displayToken = (token: string) => token.replace('</w>', '').trim() || token

export function TokenTreeMatrixCard() {
  const [summary, setSummary] = useState<MatrixSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setFailed(false)
    try {
      setSummary(await tokenTreeController.getMatrixSummary(TOP_K))
    } catch {
      setFailed(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const energyRows = (rows: [string, number, number][]) => (
    <div className="divide-y divide-border/30">
      {rows.map(([token, id, norm], i) => (
        <div key={`${id}-${i}`} className="flex items-center justify-between py-1 text-xs font-mono">
          <span className="text-muted-foreground">
            <span className="text-muted-foreground/50 w-6 inline-block tabular-nums">#{i + 1}</span>
            {displayToken(token)}
            <span className="text-muted-foreground/60 ml-1">id {id}</span>
          </span>
          <span className="text-primary tabular-nums">{norm.toFixed(4)}</span>
        </div>
      ))}
    </div>
  )

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Embedding Matrix Overview</CardTitle>
        <Button size="sm" variant="ghost" onClick={load} disabled={loading} aria-label="Refresh matrix overview">
          <IconRefresh className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && summary === null ? (
          <div className="space-y-1">
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
          </div>
        ) : failed ? (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-3 py-2 text-sm text-destructive">
            Could not load the embedding matrix overview.
          </div>
        ) : summary && summary.matrix === null ? (
          <div className="text-center py-6 text-sm text-muted-foreground">
            Embeddings are disabled for this tree. Train with embed-dim &gt; 0 to generate an embedding matrix.
          </div>
        ) : summary ? (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Chip label={`${summary.matrix?.[0]} x ${summary.matrix?.[1]} matrix`} />
              <Chip label={`norm ${summary.norm_min.toFixed(3)}–${summary.norm_max.toFixed(3)}`} />
              <Chip label={`mean ${summary.norm_mean.toFixed(3)}`} />
              <Chip label={`${summary.live_tokens} live / ${summary.dead_tokens} dead`} />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Most energetic
                </div>
                {summary.most_energetic.length ? energyRows(summary.most_energetic) : null}
              </div>
              <div className="space-y-1">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Least energetic
                </div>
                {summary.least_energetic.length ? energyRows(summary.least_energetic) : null}
              </div>
            </div>

            <p className="text-xs text-muted-foreground">
              Each row of the embedding matrix is the L2-normalized vector generated from a token&apos;s compressed
              pugqeep point. Norm &quot;energy&quot; ranks tokens by how much signal their point reconstructs; dead
              tokens reconstruct to zero.
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
