'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Skeleton, Chip } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import {
  tokenTreeController,
  type VocabPage,
  type VocabEntry,
  type LineageResult,
  type TokenTreeStats,
} from '@/lib/token-tree-controller'

interface TokenTreeVocabCardProps {
  refreshKey?: number
}

const display = (token: string) => token.replace('</w>', '').trim() || token

const PAGE_SIZE = 50

export function TokenTreeVocabCard({ refreshKey = 0 }: TokenTreeVocabCardProps) {
  const [stats, setStats] = useState<TokenTreeStats | null>(null)
  const [page, setPage] = useState<VocabPage | null>(null)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [expanded, setExpanded] = useState<VocabEntry | null>(null)
  const [lineage, setLineage] = useState<LineageResult | null>(null)
  const [lineageLoading, setLineageLoading] = useState(false)

  useEffect(() => {
    let active = true
    tokenTreeController.getStats().then(s => { if (active) setStats(s) }).catch(() => { if (active) setStats(null) })
    return () => { active = false }
  }, [])

  const load = useCallback(async (nextOffset: number) => {
    setLoading(true)
    try {
      setPage(await tokenTreeController.getVocab(PAGE_SIZE, nextOffset))
      setLoadFailed(false)
    } catch {
      setPage(null)
      setLoadFailed(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (stats?.trained) load(offset)
  }, [refreshKey, stats, offset, load])

  const handleGoTo = async (nextOffset: number) => {
    setOffset(nextOffset)
    setExpanded(null)
    setLineage(null)
    await load(nextOffset)
  }

  const handleToggleEntry = async (entry: VocabEntry) => {
    if (expanded?.id === entry.id) {
      setExpanded(null)
      setLineage(null)
      return
    }
    setExpanded(entry)
    setLineage(null)
    setLineageLoading(true)
    try {
      setLineage(await tokenTreeController.lineage(entry.token))
    } catch {
      setLineage(null)
    } finally {
      setLineageLoading(false)
    }
  }

  const trained = stats?.trained ?? false

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Token Tree Vocabulary</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => load(offset)} disabled={loading} aria-label="Refresh vocabulary">
          <IconRefresh className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {stats ? (
          <div className="flex flex-wrap gap-2">
            <Chip label={`Vocab ${stats.vocab_size}`} />
            <Chip label={`Merged ${stats.num_merges}`} />
            <Chip label={`Base ${stats.num_base_tokens}`} />
          </div>
        ) : (
          <div className="flex gap-2">
            <Skeleton className="h-6 w-24 rounded" />
            <Skeleton className="h-6 w-24 rounded" />
          </div>
        )}

        {!trained || page === null ? (
          <div className="text-center py-6 text-sm text-muted-foreground">
            {loadFailed
              ? 'Could not load the vocabulary. Refresh to try again.'
              : stats === null
                ? 'Could not load the token tree.'
                : 'The token tree is not trained. Train it to browse its vocabulary.'}
          </div>
        ) : loading && page === null ? (
          <div className="space-y-1">
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {page.entries.map(entry => (
              <div key={entry.id}>
                <button
                  type="button"
                  onClick={() => handleToggleEntry(entry)}
                  className="w-full flex items-center gap-3 py-1.5 text-sm text-left hover:bg-muted/40 transition-colors px-1 -mx-1 rounded"
                  aria-label={`Toggle lineage for ${display(entry.token)}`}
                >
                  <span className="w-10 text-right text-xs text-muted-foreground tabular-nums">{entry.id}</span>
                  <span className={`font-mono flex-1 ${entry.is_special ? 'text-primary' : ''}`}>
                    {display(entry.token)}
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums">{entry.freq}</span>
                  {entry.is_special && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-primary/10 text-primary">Special</span>
                  )}
                  {entry.is_merged && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-muted text-muted-foreground">Merged</span>
                  )}
                </button>
                {expanded?.id === entry.id && (
                  <div className="rounded-md bg-muted/50 px-3 py-2 my-1">
                    <div className="text-xs text-muted-foreground mb-1">
                      Merge lineage of <span className="font-mono text-primary">"{display(entry.token)}"</span>
                      <span className="text-muted-foreground/70"> — {lineage?.leaves.length ?? '…'} character leaves</span>
                    </div>
                    {lineageLoading ? (
                      <Skeleton className="h-12 w-full rounded" />
                    ) : lineage ? (
                      <div className="space-y-1">
                        <div className="flex flex-wrap gap-1">
                          {lineage.leaves.map((leaf, i) => (
                            <Chip key={i} label={leaf} />
                          ))}
                        </div>
                        <pre className="text-xs font-mono whitespace-pre-wrap break-words text-muted-foreground">
                          {lineage.tree}
                        </pre>
                      </div>
                    ) : (
                      <div className="text-xs text-muted-foreground">Lineage unavailable.</div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {page && page.entries.length > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              Showing {offset + 1}–{offset + page.entries.length} of {page.total}
            </span>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleGoTo(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0 || loading}
              >
                Prev
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleGoTo(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= page.total || loading}
              >
                Next
              </Button>
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Browse the tree's vocabulary in id order — special tokens and base characters first, then merge tokens in
          the order they were learned. Click a token to expand its lineage down to character leaves.
        </p>
      </CardContent>
    </Card>
  )
}
