'use client'

import { useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { memoryController, type MemoryItem } from '@/lib/memory-controller'

interface MemoryAddFormProps {
  onAdded: () => void
  highlightItem: (content: string, items: MemoryItem[]) => void
  items: MemoryItem[]
}

export function MemoryAddForm({ onAdded, highlightItem, items }: MemoryAddFormProps) {
  const [content, setContent] = useState('')
  const [topic, setTopic] = useState('')
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAdd = async () => {
    if (!content.trim()) return
    setAdding(true)
    setError(null)
    try {
      const result = await memoryController.store(content, topic.trim() || 'manual')
      if (result.stored) {
        const saved = content.trim()
        setContent('')
        setTopic('')
        await onAdded()
        highlightItem(saved, items)
      } else {
        setError('Already remembered (or memory is disabled)')
      }
    } catch {
      setError('Could not store fact')
    } finally {
      setAdding(false)
    }
  }

  return (
    <div className="space-y-1.5 rounded border border-border/40 p-2">
      <textarea
        value={content}
        onChange={e => setContent(e.target.value)}
        placeholder="Type a fact the AI should remember..."
        aria-label="New memory fact"
        className="w-full h-14 resize-none rounded border border-border/40 bg-background p-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
      />
      <div className="flex items-center gap-1.5">
        <input
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder="topic"
          aria-label="Memory fact topic"
          className="flex-1 h-7 rounded border border-border/40 bg-background px-2 text-[10px] focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
        />
        <Button size="sm" className="h-6 text-[10px] px-2" disabled={!content.trim() || adding} onClick={handleAdd}>
          {adding ? 'Saving…' : 'Save'}
        </Button>
      </div>
      {error && <StatusBanner variant="error" message={error} dismissible={false} />}
    </div>
  )
}
