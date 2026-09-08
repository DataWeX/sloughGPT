'use client'

import { IconTrash, IconEdit } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { memoryController, type MemoryItem } from '@/lib/memory-controller'
import { formatRelativeTime } from '@/lib/format-bytes'

interface MemoryItemListProps {
  items: MemoryItem[]
  highlightedId: string | null
  copiedId: string | null
  searchResults: MemoryItem[] | null
  onCopy: (content: string, id: string) => void
  onEdit: (item: MemoryItem) => void
  onDelete: (item: MemoryItem) => void
}

export function MemoryItemList({
  items,
  highlightedId,
  copiedId,
  searchResults,
  onCopy,
  onEdit,
  onDelete,
}: MemoryItemListProps) {
  return (
    <ul className="space-y-1 max-h-60 overflow-y-auto">
      {items.map(item => (
        <li key={item.id} className={cn(
          'group flex items-start justify-between gap-2 p-2 rounded bg-muted/30 border text-xs leading-relaxed transition-colors',
          item.id === highlightedId ? 'border-primary/60 bg-primary/10' : 'border-border/40',
        )}>
          <div className="min-w-0">
            <div className="flex items-start justify-between gap-1">
              <span
                className="block cursor-pointer select-text hover:text-foreground/80 transition-colors"
                title="Copy to clipboard"
                role="button"
                tabIndex={0}
                onClick={() => onCopy(item.content, item.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onCopy(item.content, item.id)
                  }
                }}
              >
                {item.content.length > 160 ? item.content.slice(0, 160) + '…' : item.content}
              </span>
              {copiedId === item.id && (
                <span className="shrink-0 text-[9px] text-success font-medium pt-0.5">Copied</span>
              )}
            </div>
            <div className="flex items-center gap-1.5 mt-1 flex-wrap">
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
          <div className="flex items-center gap-0.5 shrink-0">
            {typeof item.score === 'number' && searchResults !== null && (
              <span className="text-[10px] text-muted-foreground font-mono shrink-0 mr-0.5">{item.score.toFixed(2)}</span>
            )}
            <button
              type="button"
              onClick={() => onEdit(item)}
              className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 p-0.5 text-muted-foreground hover:text-primary transition-opacity"
              aria-label="Edit memory item"
            >
              <IconEdit className="h-3 w-3" />
            </button>
            <button
              type="button"
              onClick={() => onDelete(item)}
              className="opacity-0 group-hover:opacity-100 focus-within:opacity-100 p-0.5 text-muted-foreground hover:text-destructive transition-opacity"
              aria-label="Delete memory item"
            >
              <IconTrash className="h-3 w-3" />
            </button>
          </div>
        </li>
      ))}
    </ul>
  )
}
