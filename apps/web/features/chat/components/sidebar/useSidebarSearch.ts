'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import type { Conversation } from '@/lib/session-controller'
import { sessionController } from '@/lib/session-controller'
import { MS_PER_DAY } from '@/lib/format-bytes'

const SORT_KEY = 'sloughgpt:sidebar-sort'

export interface UseSidebarSearchReturn {
  search: string
  setSearch: (q: string) => void
  sortMode: 'updated' | 'name' | 'messages'
  setSortMode: (m: 'updated' | 'name' | 'messages') => void
  sortOpen: boolean
  setSortOpen: (v: boolean) => void
  serverSearchLoading: boolean
  filtered: Conversation[]
  starred: Conversation[]
  pinned: Conversation[]
  recencyGroups: { label: string; conversations: Conversation[] }[]
  q: string
}

export function useSidebarSearch(conversations: Conversation[]): UseSidebarSearchReturn {
  const [search, setSearch] = useState('')
  const [serverSearchResults, setServerSearchResults] = useState<Conversation[] | null>(null)
  const [serverSearchLoading, setServerSearchLoading] = useState(false)
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [sortMode, setSortMode] = useState<'updated' | 'name' | 'messages'>(() => {
    if (typeof window === 'undefined') return 'updated'
    const saved = localStorage.getItem(SORT_KEY)
    if (saved === 'name' || saved === 'messages') return saved
    return 'updated'
  })
  const [sortOpen, setSortOpen] = useState(false)

  useEffect(() => {
    localStorage.setItem(SORT_KEY, sortMode)
  }, [sortMode])

  const sorted = useMemo(() => {
    return [...conversations].sort((a, b) => {
      if (sortMode === 'name') {
        return (a.name || '').localeCompare(b.name || '')
      }
      if (sortMode === 'messages') {
        return (b.message_count ?? b.messages?.length ?? 0) - (a.message_count ?? a.messages?.length ?? 0)
      }
      return new Date(b.updated_at || b.updatedAt || 0).getTime() - new Date(a.updated_at || a.updatedAt || 0).getTime()
    })
  }, [conversations, sortMode])

  const q = search.toLowerCase().trim()

  useEffect(() => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    if (!q || q.length < 2) {
      setServerSearchResults(null)
      setServerSearchLoading(false)
      return
    }
    setServerSearchLoading(true)
    searchDebounceRef.current = setTimeout(async () => {
      try {
        const results = await sessionController.search(q, 30)
        setServerSearchResults(results.map(r => ({
          id: r.id,
          name: r.name || 'Untitled',
          session_id: r.id,
          messages: r.matches?.map(m => ({ id: m.timestamp, role: m.role, content: m.content })) || [],
          updated_at: r.updated_at,
          created_at: r.created_at,
          pinned: false,
          starred: false,
          message_count: r.match_count,
        })))
      } catch {
        setServerSearchResults(null)
      } finally {
        setServerSearchLoading(false)
      }
    }, 300)
    return () => { if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current) }
  }, [q])

  const filtered = useMemo(() => {
    if (!q) return sorted
    if (serverSearchResults) return serverSearchResults
    return sorted.filter(c =>
      c.name?.toLowerCase().includes(q) ||
      c.messages?.some(m => m.content?.toLowerCase().includes(q))
    )
  }, [sorted, q, serverSearchResults])

  const starred = useMemo(() => filtered.filter(c => c.starred).slice(0, 10), [filtered])
  const unstarred = useMemo(() => filtered.filter(c => !c.starred), [filtered])
  const pinned = useMemo(() => unstarred.filter(c => c.pinned), [unstarred])
  const unpinned = useMemo(() => unstarred.filter(c => !c.pinned), [unstarred])

  function recencyGroup(dateStr: string | undefined): string {
    if (!dateStr) return 'Older'
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = diff / MS_PER_DAY
    if (days < 1) return 'Today'
    if (days < 2) return 'Yesterday'
    if (days < 7) return 'Last 7 days'
    return 'Older'
  }

  const recencyGroups = useMemo(() => {
    const groups: { label: string; conversations: Conversation[] }[] = []
    const seen = new Set<string>()
    for (const c of unpinned) {
      const label = recencyGroup(c.updated_at || c.updatedAt)
      if (!seen.has(label)) {
        seen.add(label)
        groups.push({ label, conversations: [] })
      }
      const group = groups.find(g => g.label === label)!
      if (group.conversations.length < 15) group.conversations.push(c)
    }
    const order = ['Today', 'Yesterday', 'Last 7 days', 'Older']
    return groups.sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label))
  }, [unpinned])

  return {
    search,
    setSearch,
    sortMode,
    setSortMode,
    sortOpen,
    setSortOpen,
    serverSearchLoading,
    filtered,
    starred,
    pinned,
    recencyGroups,
    q,
  }
}
