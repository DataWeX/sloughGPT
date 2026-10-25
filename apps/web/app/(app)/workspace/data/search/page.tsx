'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter, useSearchParams } from '@/vite/next-compat/navigation'
import { Skeleton } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { WorkspaceSectionTabs } from '@/components/workspace/WorkspaceSectionTabs'
import { dataTabs } from '@/components/workspace/workspace-tabs'
import { SearchInputCard } from '@/components/workspace-search/SearchInputCard'
import { SearchResultsCard } from '@/components/workspace-search/SearchResultsCard'
import { SearchEmptyState } from '@/components/workspace-search/SearchEmptyState'
import { apiGet } from '@/lib/http-client'
import { useAuthStore } from '@/lib/auth'
import { useToastStore } from '@/lib/toast-store'
import { SEARCH_STORES } from '@/lib/search-stores'

interface SearchHit {
  id: string
  store: string
  title: string
  detail: string
  score: number
  locator: string
}

// apiGet unwraps the {status,data} envelope — the type is the payload.
interface SearchResponse {
  hits: SearchHit[]
  partial: string[]
  skipped: string[]
  query: string
}

export default function WorkspaceSearchPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  // Deep link: /workspace/data/search?q=... seeds the query (palette jumps).
  const [query, setQuery] = useState(() => searchParams.get('q') ?? '')
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [partial, setPartial] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const search = useCallback(async (q: string) => {
    if (!currentWorkspace?.id || !q.trim()) {
      setHits(null)
      setPartial([])
      return
    }
    setLoading(true)
    try {
      const params = new URLSearchParams({ q: q.trim(), workspace_id: currentWorkspace.id })
      const res = await apiGet<SearchResponse>(`/search?${params.toString()}`)
      if (res?.hits) {
        setHits(res.hits)
        setPartial(res.partial ?? [])
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
      else { setHits(null); setPartial([]) }
    }, 300)
    return () => clearTimeout(timer)
  }, [query, search])

  // Grouping by store is presentation — the core returns one ranked list.
  type GroupItem = { id: string; type: string; title: string; detail: string }
  type Group = { type: string; label: string; color: string; link: string; items: GroupItem[] }
  const groups: Group[] = []
  if (hits) {
    const byStore = new Map<string, SearchHit[]>()
    for (const hit of hits) {
      const bucket = byStore.get(hit.store)
      if (bucket) bucket.push(hit)
      else byStore.set(hit.store, [hit])
    }
    for (const [store, items] of byStore) {
      const meta = SEARCH_STORES[store]
      if (meta) {
        groups.push({
          type: store,
          label: meta.label,
          color: meta.color,
          link: meta.link,
          items: items.map(h => ({ id: h.id, type: store, title: h.title, detail: h.detail })),
        })
      }
    }
  }

  return (
    <PageContainer
      title="Workspace Search"
      toolbar={<WorkspaceSectionTabs tabs={dataTabs} ariaLabel="Workspace data" />}
    >
      <SearchInputCard query={query} onQueryChange={setQuery} />

      {loading && <Skeleton className="h-32 w-full" />}

      {!loading && !hits && <SearchEmptyState />}

      {!loading && hits && (
        <>
          {partial.length > 0 && (
            <p className="text-xs text-warning mb-2" role="status">
              Partial results — some sources are unavailable: {partial.join(', ')}
            </p>
          )}
          <SearchResultsCard
            groups={groups}
            total={hits.length}
            query={query}
            onNavigate={link => router.push(link)}
          />
        </>
      )}
    </PageContainer>
  )
}
