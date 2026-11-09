'use client'
import { notify } from '@/lib/notify'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  memoryController,
  type MemoryItem,
  type MemoryStats,
  type MemoryArchiveStats,
  type MemoryArchiveRecord,
} from '@/lib/memory-controller'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export interface UseMemoryDataReturn {
  stats: MemoryStats | null
  items: MemoryItem[]
  archiveStats: MemoryArchiveStats | null
  loading: boolean
  searched: boolean
  searchResults: MemoryItem[] | null
  setSearch: (q: string) => void
  search: string
  fetchData: () => Promise<void>
  handleSearch: (q: string) => Promise<void>
  setSearchResults: (v: MemoryItem[] | null) => void
  setSearched: (v: boolean) => void
  setItems: (v: MemoryItem[] | ((prev: MemoryItem[]) => MemoryItem[])) => void
}

export function useMemoryData(): UseMemoryDataReturn {
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [items, setItems] = useState<MemoryItem[]>([])
  const [archiveStats, setArchiveStats] = useState<MemoryArchiveStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [searchResults, setSearchResults] = useState<MemoryItem[] | null>(null)
  const [searched, setSearched] = useState(false)

  const searchRef = useRef(search)
  searchRef.current = search

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [statsResult, listResult, archiveResult] = await Promise.all([
        memoryController.stats().catch(() => null),
        memoryController.list(),
        memoryController.archiveStats().catch(() => null),
      ])
      setStats(statsResult)
      setItems(listResult.items || [])
      setArchiveStats(archiveResult)
      const q = searchRef.current.trim()
      if (q) {
        const searchResult = await memoryController.search(q)
        setSearchResults(searchResult.results || [])
      }
    } catch {
      notify.toast('Could not load memory', { type: 'error' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const [statsResult, listResult, archiveResult] = await Promise.all([
          memoryController.stats().catch(() => null),
          memoryController.list(),
          memoryController.archiveStats().catch(() => null),
        ])
        if (active) {
          setStats(statsResult)
          setItems(listResult.items || [])
          setArchiveStats(archiveResult)
        }
      } catch {
        if (active) notify.toast('Could not load memory', { type: 'error' })
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => {
      active = false
    }
  }, [])

  const handleSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSearchResults(null)
      setSearched(false)
      return
    }
    try {
      const result = await memoryController.search(q)
      setSearchResults(result.results || [])
      setSearched(true)
    } catch {
      notify.toast('Could not memory search', { type: 'error' })
    }
  }, [])

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      handleSearch(search)
    }, 300)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [search, handleSearch])

  return {
    stats,
    items,
    archiveStats,
    loading,
    searched,
    searchResults,
    setSearch,
    search,
    fetchData,
    handleSearch,
    setSearchResults,
    setSearched,
    setItems,
  }
}
