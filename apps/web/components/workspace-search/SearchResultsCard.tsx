'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import { ExternalLink } from 'lucide-react'

interface SearchResult {
  id: string
  type: string
  title: string
  detail: string
}

interface SearchResultGroup {
  type: string
  label: string
  color: string
  link: string
  items: SearchResult[]
}

interface SearchResultsCardProps {
  groups: SearchResultGroup[]
  total: number
  query: string
  onNavigate?: (link: string) => void
}

export function SearchResultsCard({ groups, total, query, onNavigate }: SearchResultsCardProps) {
  if (total === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">No results for &ldquo;{query}&rdquo;</p>
    )
  }

  return (
    <>
      <p className="text-sm text-muted-foreground mb-4">{total} result(s) found</p>
      {groups.map((group) => {
        if (group.items.length === 0) return null
        return (
          <Card key={group.type} className="mb-4">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs flex items-center gap-2">
                <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${group.color}`}>
                  {group.label}
                </span>
                <span className="text-muted-foreground">({group.items.length})</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              {group.items.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onNavigate?.(group.link)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50 transition-colors text-left group"
                >
                  <div className="min-w-0 flex-1">
                    <div className="font-medium group-hover:text-primary transition-colors">{item.title}</div>
                    {item.detail && <div className="text-muted-foreground truncate">{item.detail}</div>}
                  </div>
                  <ExternalLink className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0 ml-2" />
                </button>
              ))}
            </CardContent>
          </Card>
        )
      })}
    </>
  )
}
