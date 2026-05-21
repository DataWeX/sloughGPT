'use client'

import { useState, useEffect, useCallback } from 'react'
import { knowledgeController, type KnowledgeItem } from '@/lib/knowledge-controller'

interface UseKnowledgeResult {
  items: KnowledgeItem[]
  loading: boolean
  add: (content: string, topic?: string) => Promise<void>
  remove: (id: string) => Promise<void>
  search: (query: string) => Promise<KnowledgeItem[]>
  refresh: () => Promise<void>
  stats: {
    total: number
    topics: number
  }
}

export function useKnowledge(): UseKnowledgeResult {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [loading, setLoading] = useState(true)

  const fetchItems = useCallback(async () => {
    try {
      const data = await knowledgeController.list()
      setItems(data || [])
    } catch (err) {
      console.error('Failed to fetch knowledge:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchItems()
  }, [fetchItems])

  const add = useCallback(async (content: string, topic: string = 'injected') => {
    if (!content.trim()) return
    try {
      await knowledgeController.add(content, topic)
      await fetchItems()
    } catch (err) {
      console.error('Failed to add knowledge:', err)
    }
  }, [fetchItems])

  const remove = useCallback(async (id: string) => {
    try {
      await knowledgeController.delete(id)
      setItems(prev => prev.filter(item => item.id !== id))
    } catch (err) {
      console.error('Failed to delete knowledge:', err)
    }
  }, [])

  const search = useCallback(async (query: string): Promise<KnowledgeItem[]> => {
    try {
      const data = await knowledgeController.search(query)
      return data.results || []
    } catch (err) {
      console.error('Failed to search knowledge:', err)
    }
    return []
  }, [])

  const topics = new Set(items.map(i => i.topic))

  return {
    items,
    loading,
    add,
    remove,
    search,
    refresh: fetchItems,
    stats: {
      total: items.length,
      topics: topics.size,
    },
  }
}
