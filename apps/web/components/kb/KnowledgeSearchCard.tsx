'use client'

import { Card, CardContent, Button, Input } from '@sloughgpt/strui'
import type { KnowledgeItem } from '@/lib/kb-controller'

interface KnowledgeSearchCardProps {
  query: string
  results: KnowledgeItem[]
  loading: boolean
  onQueryChange: (value: string) => void
  onSearch: () => void
}

export function KnowledgeSearchCard({
  query,
  results,
  loading,
  onQueryChange,
  onSearch,
}: KnowledgeSearchCardProps) {
  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={e => onQueryChange(e.target.value)}
          placeholder="Search knowledge..."
          className="h-7 text-[11px]"
          onKeyDown={e => e.key === 'Enter' && onSearch()}
        />
        <Button onClick={onSearch} disabled={loading} className="shrink-0 h-7 text-[11px]">
          {loading ? 'Searching...' : 'Search'}
        </Button>
      </div>
      {results.length > 0 && (
        <div className="space-y-2">
          {results.map(item => (
            <Card key={item.id}>
              <CardContent className="py-3">
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-xs">{item.content.slice(0, 300)}{item.content.length > 300 ? '...' : ''}</p>
                    <div className="mt-1.5 flex gap-2">
                      <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] text-primary">{item.topic}</span>
                      {item.score != null && (
                        <span className="text-xs text-muted-foreground">Score: {item.score.toFixed(3)}</span>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
