'use client'

import { useCallback, useMemo } from 'react'
import { cn, Checkbox, Button, Skeleton } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconEdit } from '@sloughgpt/strui'
import { formatRelativeTime } from '@/lib/format-bytes'
import type { MemoryItem } from '@/lib/memory-controller'

interface MemoryItemListProps {
  items: MemoryItem[]
  searchResults: MemoryItem[] | null
  loading: boolean
  searched: boolean
  activeTopic: string | null
  sortOrder: 'newest' | 'oldest' | 'importance'
  showAllItems: boolean
  selectedIds: Set<string>
  onToggleSelect: (id: string) => void
  onToggleSelectAll: () => void
  onClearSelection: () => void
  onStartEdit: (item: MemoryItem) => void
  onSetPendingDelete: (item: MemoryItem) => void
  onSetPendingBatchDelete: (open: boolean) => void
  onExportSelected: () => void
  onCopy: (text: string) => void
  onClearSearch: () => void
  setShowAllItems: (v: boolean | ((v: boolean) => boolean)) => void
  setSearch: (v: string) => void
  setSearchResults: (v: MemoryItem[] | null) => void
  setSearched: (v: boolean) => void
}

export function MemoryItemList({
  items, searchResults, loading, searched, activeTopic, sortOrder,
  showAllItems, selectedIds, onToggleSelect, onToggleSelectAll,
  onClearSelection, onStartEdit, onSetPendingDelete, onSetPendingBatchDelete,
  onExportSelected, onCopy, onClearSearch, setShowAllItems,
  setSearch, setSearchResults, setSearched,
}: MemoryItemListProps) {
  const browseList = useMemo(() => {
    const base = activeTopic ? items.filter(i => i.topic === activeTopic) : items
    return [...base].sort((a, b) => {
      if (sortOrder === 'importance') return (b.importance ?? 0) - (a.importance ?? 0)
      return sortOrder === 'newest' ? b.timestamp - a.timestamp : a.timestamp - b.timestamp
    })
  }, [items, activeTopic, sortOrder])

  const filteredByTopic = useMemo(() => {
    const base = searchResults !== null ? searchResults : browseList
    if (!activeTopic) return base
    return base.filter(i => i.topic === activeTopic)
  }, [searchResults, browseList, activeTopic])

  const visibleList = showAllItems ? filteredByTopic : filteredByTopic.slice(0, 10)

  if (loading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 rounded-lg" />
        <Skeleton className="h-12 rounded-lg" />
        <Skeleton className="h-12 rounded-lg" />
      </div>
    )
  }

  if (filteredByTopic.length === 0) {
    return (
      <div className="text-center py-6 text-sm text-muted-foreground">
        {activeTopic
          ? `No memory in the "${activeTopic}" topic.`
          : (searched ? 'No memory matches that search.' : 'Nothing remembered yet. The AI stores facts automatically as you chat.')}
        {searched && !activeTopic && (
          <div className="mt-2">
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs"
              onClick={onClearSearch}
            >
              Clear search
            </Button>
          </div>
        )}
      </div>
    )
  }

  return (
    <div>
      {filteredByTopic.length > 1 && (
        <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer mb-1.5">
          <Checkbox
            checked={selectedIds.size === filteredByTopic.length}
            onCheckedChange={onToggleSelectAll}
            className="rounded border-border"
            aria-label="Select all memory facts"
          />
          Select all ({filteredByTopic.length})
        </label>
      )}
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-1.5 mb-2">
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs"
            onClick={onExportSelected}
            aria-label={`Export ${selectedIds.size} selected memory items`}
          >
            <IconDownload className="h-3 w-3 mr-1" />
            Export ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="destructive"
            className="h-8 text-xs"
            onClick={() => onSetPendingBatchDelete(true)}
            aria-label={`Delete ${selectedIds.size} selected memory items`}
          >
            <IconTrash className="h-3 w-3 mr-1" />
            Delete ({selectedIds.size})
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-8 text-xs ml-auto"
            onClick={onClearSelection}
          >
            Cancel
          </Button>
        </div>
      )}
      <div className="space-y-1.5">
        {visibleList.map(item => (
          <div
            key={item.id}
            className={cn('group flex items-start justify-between gap-2 rounded-lg border px-3 py-2 transition-colors', selectedIds.has(item.id)
                ? 'bg-primary/[0.06] border-primary/30'
                : 'border-border/60 hover:bg-muted/40')}
          >
            <Checkbox
              checked={selectedIds.has(item.id)}
              onCheckedChange={() => onToggleSelect(item.id)}
              className="mt-1 rounded border-border shrink-0"
              aria-label={`Select memory fact ${item.content}`}
            />
            <div className="min-w-0 flex-1">
            <p
              className="text-sm line-clamp-2 cursor-pointer select-text hover:text-foreground/80 transition-colors"
              title="Copy to clipboard"
              onClick={() => onCopy(item.content)}
            >
              {item.content}
            </p>
            <div className="flex items-center gap-1.5 mt-1">
              {item.topic && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">{item.topic}</span>
              )}
              {item.source && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">{item.source}</span>
              )}
              {typeof item.importance === 'number' && (
                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium" title="Importance score">
                  importance {item.importance.toFixed(1)}
                </span>
              )}
              {item.timestamp > 0 && (
                <span
                  className="text-[9px] text-muted-foreground font-mono"
                  title={new Date(item.timestamp * 1000).toLocaleString()}
                >
                  {formatRelativeTime(item.timestamp)}
                </span>
              )}
            </div>
          </div>
          {typeof item.score === 'number' && searchResults !== null && (
            <span className="text-[10px] text-muted-foreground font-mono shrink-0">{item.score.toFixed(2)}</span>
          )}
          <button
            type="button"
            onClick={() => onStartEdit(item)}
            className="h-7 w-7 shrink-0 flex items-center justify-center rounded text-muted-foreground opacity-60 lg:opacity-0 lg:group-hover:opacity-100 hover:text-primary hover:bg-primary/10 transition-all"
            aria-label="Edit memory item"
          >
            <IconEdit className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => onSetPendingDelete(item)}
            className="h-7 w-7 shrink-0 flex items-center justify-center rounded text-muted-foreground opacity-60 lg:opacity-0 lg:group-hover:opacity-100 hover:text-destructive hover:bg-destructive/10 transition-all"
            aria-label="Delete memory item"
          >
            <IconTrash className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
      </div>
      {filteredByTopic.length > 10 && (
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/60">
          <p className="text-[10px] text-muted-foreground">
            Showing {visibleList.length} of {filteredByTopic.length}
          </p>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => setShowAllItems(v => !v)}
          >
            {showAllItems ? 'Show fewer' : 'Show all'}
          </Button>
        </div>
      )}
    </div>
  )
}
