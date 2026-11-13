'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

export interface SearchResult {
  id: string
  text: string
  score: number
}

export interface VectorSearchCardProps {
  query: string
  results: SearchResult[]
  searchTime: number | null
  searching: boolean
  onQueryChange: (q: string) => void
  onSearch: () => void
}

export function VectorSearchCard({
  query,
  results,
  searchTime,
  searching,
  onQueryChange,
  onSearch,
}: VectorSearchCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Similarity Search</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <div className="flex gap-1.5">
          <Input
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            placeholder="Search for similar text..."
            className="h-7 text-[11px] flex-1"
            onKeyDown={e => e.key === 'Enter' && onSearch()}
          />
          <Button size="sm" className="h-7 text-[11px]" onClick={onSearch} disabled={searching || !query.trim()}>
            {searching ? 'Searching...' : 'Search'}
          </Button>
        </div>

        {results.length > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] text-muted-foreground/60">{results.length} results in {searchTime?.toFixed(1)}ms</p>
            {results.map((r, i) => (
              <div key={r.id || i} className="border border-border/40 rounded-lg p-2.5 space-y-0.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px]">{r.text}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-primary/10 text-primary tabular-nums">
                    {(r.score * 100).toFixed(1)}%
                  </span>
                </div>
                {r.id && <p className="text-[10px] text-muted-foreground/60 font-mono">{r.id}</p>}
              </div>
            ))}
          </div>
        )}

        {results.length === 0 && !searching && query && (
          <p className="text-[10px] text-muted-foreground/60 text-center py-3">No results found</p>
        )}
      </CardContent>
    </Card>
  )
}
