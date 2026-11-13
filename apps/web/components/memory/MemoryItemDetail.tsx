'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface MemoryItem {
  id: string
  topic?: string
  content: string
  source?: string
  importance?: number
  timestamp?: string
}

interface MemoryItemDetailProps {
  item: MemoryItem | null
  editMode?: boolean
  editContent?: string
  onEditContentChange?: (c: string) => void
  onToggleEdit?: () => void
  onSave?: () => void
  onDelete?: (id: string) => void
}

export function MemoryItemDetail({ item, editMode, editContent, onEditContentChange, onToggleEdit, onSave, onDelete }: MemoryItemDetailProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-[11px] font-medium">{item ? (item.topic ?? 'Detail') : 'Select item'}</CardTitle>
          {item && (
            <div className="flex items-center gap-1">
              {editMode ? (
                <>
                  <Button size="sm" className="h-6 text-[10px]" onClick={onSave}>Save</Button>
                  <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={onToggleEdit}>Cancel</Button>
                </>
              ) : (
                <>
                  <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={onToggleEdit}>Edit</Button>
                  <Button size="sm" variant="ghost" className="h-6 text-[10px] text-destructive" onClick={() => onDelete?.(item.id)}>Delete</Button>
                </>
              )}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-2.5 pb-2.5">
        {item ? (
          editMode ? (
            <textarea
              value={editContent ?? ''}
              onChange={e => onEditContentChange?.(e.target.value)}
              rows={12}
              aria-label="Edit memory content"
              className="w-full rounded-lg border border-border/40 bg-background p-2.5 text-[11px]"
            />
          ) : (
            <div className="space-y-2 text-[11px]">
              <div className="flex gap-3 text-[10px] text-muted-foreground/60">
                <span>Topic: {item.topic ?? '--'}</span>
                <span>Importance: {((item.importance ?? 0) * 100).toFixed(0)}%</span>
              </div>
              {item.source && <p className="text-[10px] text-muted-foreground/60">Source: {item.source}</p>}
              {item.timestamp && <p className="text-[10px] text-muted-foreground/60">Created: {new Date(item.timestamp).toLocaleString()}</p>}
              <p className="whitespace-pre-wrap text-[11px]">{item.content}</p>
            </div>
          )
        ) : (
          <p className="text-[10px] text-muted-foreground/60">Click a memory item to view details.</p>
        )}
      </CardContent>
    </Card>
  )
}
