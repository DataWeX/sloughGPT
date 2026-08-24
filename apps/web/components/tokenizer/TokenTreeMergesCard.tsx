'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Skeleton, Chip } from '@sloughgpt/strui'
import { IconRefresh, IconSearch } from '@sloughgpt/strui'
import { tokenTreeController, type MergeRule, type LineageResult } from '@/lib/token-tree-controller'

interface TokenTreeMergesCardProps {
  refreshKey?: number
}

const display = (token: string) => token.replace('</w>', '').trim() || token

const PAGE_SIZE = 20

export function TokenTreeMergesCard({ refreshKey = 0 }: TokenTreeMergesCardProps) {
  const [merges, setMerges] = useState<MergeRule[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(PAGE_SIZE)
  const [expanded, setExpanded] = useState<MergeRule | null>(null)
  const [lineage, setLineage] = useState<LineageResult | null>(null)
  const [lineageLoading, setLineageLoading] = useState(false)

  const load = useCallback(async (nextLimit: number, nextQuery: string) => {
    setLoading(true)
    try {
      setMerges(await tokenTreeController.getMerges(nextLimit, nextQuery))
    } catch {
      setMerges([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load(limit, query)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey])

  const handleSearch = async () => {
    setExpanded(null)
    setLineage(null)
    setLimit(PAGE_SIZE)
    await load(PAGE_SIZE, query.trim())
  }

  const handleLoadMore = async () => {
    const next = limit + PAGE_SIZE
    setLimit(next)
    await load(next, query.trim())
  }

  const handleToggleRule = async (rule: MergeRule) => {
    if (expanded?.rank === rule.rank) {
      setExpanded(null)
      setLineage(null)
      return
    }
    setExpanded(rule)
    setLineage(null)
    setLineageLoading(true)
    try {
      setLineage(await tokenTreeController.lineage(rule.token))
    } catch {
      setLineage(null)
    } finally {
      setLineageLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Merge Rules Explorer</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => load(limit, query.trim())} disabled={loading} aria-label="Refresh merge rules">
          <IconRefresh className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1 max-w-xs">
            <IconSearch className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') handleSearch()
              }}
              placeholder="Filter rules, e.g. 'the'"
              className="pl-7"
              aria-label="Filter merge rules"
            />
          </div>
          <Button size="sm" variant="outline" onClick={handleSearch} disabled={loading}>
            Search
          </Button>
        </div>

        {loading && merges === null ? (
          <div className="space-y-1">
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
            <Skeleton className="h-6 w-full rounded" />
          </div>
        ) : merges === null || merges.length === 0 ? (
          <div className="text-center py-6 text-sm text-muted-foreground">
            {query.trim()
              ? `No merge rules match "${query.trim()}".`
              : 'No merge rules yet. Train the token tree to learn them.'}
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {merges.map(m => (
              <div key={m.rank}>
                <button
                  type="button"
                  onClick={() => handleToggleRule(m)}
                  className="w-full flex items-center gap-3 py-1.5 text-sm text-left hover:bg-muted/40 transition-colors px-1 -mx-1 rounded"
                  aria-label={`Toggle lineage for ${display(m.token)}`}
                >
                  <span className="w-8 text-right text-xs text-muted-foreground tabular-nums">{m.rank}</span>
                  <span className="font-mono flex-1">
                    <span className="text-muted-foreground">{display(m.left)}</span>
                    <span className="text-muted-foreground/50 mx-1">+</span>
                    <span className="text-muted-foreground">{display(m.right)}</span>
                    <span className="text-primary mx-2">→</span>
                    <span className="text-primary">{display(m.token)}</span>
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums">{m.count}</span>
                </button>
                {expanded?.rank === m.rank && (
                  <div className="rounded-md bg-muted/50 px-3 py-2 my-1">
                    <div className="text-xs text-muted-foreground mb-1">
                      Merge lineage of <span className="font-mono text-primary">"{display(m.token)}"</span>
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

        {merges && merges.length > 0 && (
          <Button size="sm" variant="outline" onClick={handleLoadMore} disabled={loading}>
            {loading ? 'Loading...' : `Show ${PAGE_SIZE} more`}
          </Button>
        )}

        <p className="text-xs text-muted-foreground">
          The most frequent BPE merges the tree learned from its corpus, ranked by count. Search filters rules by
          their parts; click a rule to expand its lineage down to character leaves.
        </p>
      </CardContent>
    </Card>
  )
}
