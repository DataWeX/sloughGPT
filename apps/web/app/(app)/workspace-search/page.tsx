'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Input, Skeleton, Badge } from '@sloughgpt/strui'
import { IconSearch } from '@/components/icons/NavIcons'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'

interface SearchResult {
  id: string
  type: string
  title: string
  detail: string
}

interface SearchResponse {
  data: {
    results: {
      members: SearchResult[]
      training_jobs: SearchResult[]
      datasets: SearchResult[]
      knowledge: SearchResult[]
    }
    total: number
  }
}

const TYPE_COLORS: Record<string, string> = {
  member: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
  training: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  dataset: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  knowledge: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
}

const TYPE_LABELS: Record<string, string> = {
  member: 'Members',
  training: 'Training Jobs',
  dataset: 'Datasets',
  knowledge: 'Knowledge',
}

export default function WorkspaceSearchPage() {
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResponse['data']['results'] | null>(null)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)

  const search = useCallback(async (q: string) => {
    if (!currentWorkspace?.id || !q.trim()) {
      setResults(null)
      setTotal(0)
      return
    }
    setLoading(true)
    try {
      const res = await apiGet<SearchResponse>(`/workspaces/${currentWorkspace.id}/search`)
      if (res?.data) {
        // Client-side filtering since backend returns all data
        const allResults = res.data.results
        const filtered: typeof allResults = { members: [], training_jobs: [], datasets: [], knowledge: [] }
        const lowerQ = q.toLowerCase()

        for (const [type, items] of Object.entries(allResults)) {
          for (const item of items) {
            if (
              item.title.toLowerCase().includes(lowerQ) ||
              item.detail.toLowerCase().includes(lowerQ)
            ) {
              filtered[type as keyof typeof filtered].push(item)
            }
          }
        }

        setResults(filtered)
        setTotal(Object.values(filtered).reduce((a, b) => a + b.length, 0))
      }
    } catch {
      addToast('Search failed', 'error')
    } finally {
      setLoading(false)
    }
  }, [currentWorkspace?.id, addToast])

  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.trim()) search(query)
      else { setResults(null); setTotal(0) }
    }, 300)
    return () => clearTimeout(timer)
  }, [query, search])

  return (
    <PageContainer title="Workspace Search">
      <AppRouteHeader left={<AppRouteHeaderLead title="Workspace Search" />} />

      <Card className="mb-6">
        <CardContent className="py-3">
          <div className="relative">
            <IconSearch className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search members, training jobs, datasets, knowledge..."
              className="pl-9"
              autoFocus
            />
          </div>
        </CardContent>
      </Card>

      {loading && (
        <Skeleton className="h-32 w-full" />
      )}

      {!loading && results && (
        <>
          <p className="text-sm text-muted-foreground mb-4">{total} result(s) found</p>

          {Object.entries(results).map(([type, items]) => (
            items.length > 0 && (
              <Card key={type} className="mb-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-xs flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${TYPE_COLORS[type]}`}>
                      {TYPE_LABELS[type]}
                    </span>
                    <span className="text-muted-foreground">({items.length})</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1">
                  {items.map(item => (
                    <div key={item.id} className="flex items-center justify-between px-3 py-2 rounded-md text-[10px] hover:bg-muted/50 border border-transparent hover:border-border/50">
                      <div className="min-w-0 flex-1">
                        <div className="font-medium">{item.title}</div>
                        {item.detail && <div className="text-muted-foreground">{item.detail}</div>}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )
          ))}

          {total === 0 && (
            <p className="text-sm text-muted-foreground text-center py-8">
              No results for &ldquo;{query}&rdquo;
            </p>
          )}
        </>
      )}

      {!loading && !results && (
        <p className="text-sm text-muted-foreground text-center py-8">
          Type to search across all workspace data
        </p>
      )}
    </PageContainer>
  )
}
