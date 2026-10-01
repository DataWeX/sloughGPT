'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
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
    query: string
    partial: string[]
  }
}

// Keys MUST match the result-category keys the API returns
// (members / training_jobs / datasets / knowledge), not item.type.
const CATEGORY_CONFIG: Record<string, { label: string; color: string; link: string }> = {
  members: {
    label: 'Members',
    color: 'bg-primary/15 text-primary dark:bg-primary/10 dark:text-primary',
    link: '/workspace/members',
  },
  training_jobs: {
    label: 'Training Jobs',
    color: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
    link: '/training/queue',
  },
  datasets: {
    label: 'Datasets',
    color: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
    link: '/datasets',
  },
  knowledge: {
    label: 'Knowledge',
    color: 'bg-accent text-accent dark:bg-accent/30 dark:text-accent',
    link: '/knowledge',
  },
}

export default function WorkspaceSearchPage() {
  const router = useRouter()
  const { currentWorkspace } = useAuthStore()
  const addToast = useToastStore(s => s.addToast)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResponse['data']['results'] | null>(null)
  const [total, setTotal] = useState(0)
  const [partial, setPartial] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const search = useCallback(async (q: string) => {
    if (!currentWorkspace?.id || !q.trim()) {
      setResults(null)
      setTotal(0)
      setPartial([])
      return
    }
    setLoading(true)
    try {
      const res = await apiGet<SearchResponse>(
        `/workspaces/${currentWorkspace.id}/search?q=${encodeURIComponent(q.trim())}`
      )
      if (res?.data) {
        setResults(res.data.results)
        setTotal(res.data.total)
        setPartial(res.data.partial ?? [])
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
      else { setResults(null); setTotal(0); setPartial([]) }
    }, 300)
    return () => clearTimeout(timer)
  }, [query, search])

  const groups = results
    ? Object.entries(results)
        .filter(([type]) => CATEGORY_CONFIG[type])
        .map(([type, items]) => ({
          type,
          label: CATEGORY_CONFIG[type].label,
          color: CATEGORY_CONFIG[type].color,
          link: CATEGORY_CONFIG[type].link,
          items,
        }))
    : []

  return (
    <PageContainer
      title="Workspace Search"
      toolbar={<WorkspaceSectionTabs tabs={dataTabs} ariaLabel="Workspace data" />}
    >
      <SearchInputCard query={query} onQueryChange={setQuery} />

      {loading && <Skeleton className="h-32 w-full" />}

      {!loading && !results && <SearchEmptyState />}

      {!loading && results && (
        <>
          {partial.length > 0 && (
            <p className="text-xs text-warning mb-2" role="status">
              Partial results — some sources are unavailable: {partial.join(', ')}
            </p>
          )}
          <SearchResultsCard
            groups={groups}
            total={total}
            query={query}
            onNavigate={link => router.push(link)}
          />
        </>
      )}
    </PageContainer>
  )
}
