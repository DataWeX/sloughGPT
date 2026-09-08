'use client'

import { useState } from 'react'
import { Button, Slider } from '@sloughgpt/strui'
import { IconEdit } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { memoryController, type MemoryItem } from '@/lib/memory-controller'

interface MemoryEditFormProps {
  item: MemoryItem
  onSaved: () => void
  onCancelled: () => void
}

export function MemoryEditForm({ item, onSaved, onCancelled }: MemoryEditFormProps) {
  const [content, setContent] = useState(item.content)
  const [topic, setTopic] = useState(item.topic || '')
  const [importance, setImportance] = useState(typeof item.importance === 'number' ? item.importance : 0.5)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSave = async () => {
    if (!content.trim()) return
    setSaving(true)
    setError(null)
    try {
      const result = await memoryController.update(item.id, content, topic, importance)
      if (result.updated > 0) {
        onSaved()
      } else if (result.duplicate) {
        setError('That fact already exists in memory')
      } else {
        setError('Memory item not found')
      }
    } catch {
      setError('Could not update memory item')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-1.5 rounded border border-primary/40 p-2">
      <p className="text-[10px] font-medium text-muted-foreground flex items-center gap-1.5">
        <IconEdit className="h-3 w-3" />
        Edit memory fact
      </p>
      <textarea
        value={content}
        onChange={e => setContent(e.target.value)}
        aria-label="Edit memory fact text"
        className="w-full h-14 resize-none rounded border border-border/40 bg-background p-2 text-xs focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
      />
      <div className="flex items-center gap-1.5">
        <input
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder={item.topic || 'topic'}
          aria-label="Edit memory fact topic"
          className="flex-1 h-7 rounded border border-border/40 bg-background px-2 text-[10px] focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
        />
      </div>
      <div className="flex items-center gap-1.5">
        <Slider
          label="Importance"
          value={[importance]}
          min={0}
          max={1}
          step={0.1}
          showValue
          formatValue={(v) => v.toFixed(1)}
          onValueChange={([v]) => setImportance(v)}
          size="sm"
          className="flex-1"
          aria-label="Edit memory fact importance"
        />
      </div>
      <div className="flex items-center gap-1.5">
        <Button size="sm" className="h-6 text-[10px] px-2" disabled={!content.trim() || saving} onClick={handleSave}>
          {saving ? 'Saving…' : 'Save'}
        </Button>
        <Button size="sm" variant="ghost" className="h-6 text-[10px] px-2" onClick={onCancelled}>
          Cancel
        </Button>
      </div>
      {error && <StatusBanner variant="error" message={error} dismissible={false} />}
    </div>
  )
}
