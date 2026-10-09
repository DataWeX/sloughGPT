'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

export interface VectorAddEntriesCardProps {
  text: string
  loading: boolean
  onTextChange: (t: string) => void
  onAdd: () => void
}

export function VectorAddEntriesCard({
  text,
  loading,
  onTextChange,
  onAdd,
}: VectorAddEntriesCardProps) {
  const lineCount = text.split('\n').filter(t => t.trim()).length

  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Add Entries</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <p className="text-[10px] text-muted-foreground/60">
          Add text entries to the vector store. Each line becomes one entry.
        </p>
        <textarea
          value={text}
          onChange={e => onTextChange(e.target.value)}
          placeholder="Enter text entries, one per line..."
          className="w-full h-20 text-[11px] font-mono rounded-lg border border-border/40 bg-background px-2.5 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-ring"
          aria-label="Vector store entries"
        />
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground/60 font-mono tabular-nums">{lineCount} entries</span>
          <Button size="sm" className="h-7 text-[11px]" onClick={onAdd} disabled={loading || !text.trim()}>
            {loading ? 'Adding...' : 'Add entries'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
