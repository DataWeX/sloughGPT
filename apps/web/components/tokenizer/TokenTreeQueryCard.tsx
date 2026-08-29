'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Button, Chip, Skeleton } from '@sloughgpt/strui'
import { IconSearch } from '@sloughgpt/strui'
import { tokenTreeController, type TokenTreeStats, type SimilarResult, type Neighbor, type LineageResult } from '@/lib/token-tree-controller'
import { useToastStore } from '@/lib/toast-store'

const displayToken = (token: string) => token.replace('</w>', '').trim()

export function TokenTreeQueryCard() {
  const [stats, setStats] = useState<TokenTreeStats | null>(null)
  const [query, setQuery] = useState('quick')
  const [result, setResult] = useState<SimilarResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [expanded, setExpanded] = useState<Neighbor | null>(null)
  const [lineage, setLineage] = useState<LineageResult | null>(null)
  const [loadingLineage, setLoadingLineage] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    let active = true
    tokenTreeController.getStats().then(s => { if (active) setStats(s) }).catch(() => { if (active) setStats(null) })
    return () => { active = false }
  }, [])

  const handleSearch = async () => {
    const term = query.trim()
    if (!term) return
    setSearching(true)
    setResult(null)
    setExpanded(null)
    setLineage(null)
    try {
      setResult(await tokenTreeController.similar(term, 5))
    } catch {
      addToast(`Token not found: ${term}`, 'error')
    } finally {
      setSearching(false)
    }
  }

  const handleExpand = async (n: Neighbor) => {
    if (expanded?.id === n.id) {
      setExpanded(null)
      setLineage(null)
      return
    }
    setExpanded(n)
    setLineage(null)
    setLoadingLineage(true)
    try {
      setLineage(await tokenTreeController.lineage(String(n.id)))
    } catch {
      setLineage(null)
    } finally {
      setLoadingLineage(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Token Tree Semantic Queries</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {stats ? (
          <div className="flex flex-wrap gap-2">
            <Chip label={`Vocab ${stats.vocab_size}`} />
            <Chip label={`Embed dim ${stats.embed_dim}`} />
            <Chip label={`Points ${stats.embedding_points}`} />
            <Chip label={`Compression ${stats.embedding_compression_ratio}x`} />
          </div>
        ) : (
          <div className="flex gap-2">
            <Skeleton className="h-6 w-20 rounded" />
            <Skeleton className="h-6 w-24 rounded" />
            <Skeleton className="h-6 w-24 rounded" />
          </div>
        )}

        <div className="flex items-center gap-2">
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleSearch()
            }}
            placeholder="Query a token, e.g. quick"
            className="max-w-xs"
            aria-label="Token to query"
          />
          <Button size="sm" onClick={handleSearch} disabled={searching || !query.trim()}>
            {searching ? 'Searching...' : (
              <>
                <IconSearch className="h-4 w-4 mr-1" />
                Find neighbors
              </>
            )}
          </Button>
        </div>

        {result && (
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Nearest neighbors of{' '}
              <span className="font-mono text-primary">"{displayToken(result.query)}"</span>
            </div>
            {result.neighbors.length === 0 ? (
              <div className="text-sm text-muted-foreground">No neighbors found.</div>
            ) : (
              <div className="space-y-1">
                {result.neighbors.map(n => (
                  <button
                    key={n.id}
                    type="button"
                    onClick={() => handleExpand(n)}
                    className="w-full flex items-center gap-3 text-sm px-2 py-1.5 rounded hover:bg-muted/50 transition-colors text-left"
                    aria-label={`Toggle lineage for ${displayToken(n.token)}`}
                  >
                    <span className="font-mono w-20 truncate">{displayToken(n.token)}</span>
                    <span className="flex-1 h-1.5 bg-muted rounded overflow-hidden">
                      <span
                        className="block h-full bg-primary/70 rounded"
                        style={{ width: `${Math.round(n.score * 100)}%` }}
                      />
                    </span>
                    <span className="text-xs text-muted-foreground w-10 text-right">{n.score.toFixed(3)}</span>
                  </button>
                ))}
                {expanded && (
                  <div className="rounded-md bg-muted/50 px-3 py-2">
                    <div className="text-xs text-muted-foreground mb-1">
                      Merge lineage of <span className="font-mono text-primary">"{displayToken(expanded.token)}"</span>
                    </div>
                    {loadingLineage ? (
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
            )}
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Embeddings are learned from token co-occurrence and stored as compressed cluster points — the same
          pugqeep substrate used to generate model weights.
        </p>
      </CardContent>
    </Card>
  )
}
