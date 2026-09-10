'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface SavedTree {
  name: string
  vocab_size: number
  num_merges: number
}

interface SavedTreesCardProps {
  savedTrees: SavedTree[]
  saveName: string
  loading: boolean
  onSaveNameChange: (name: string) => void
  onSave: () => void
  onLoad: (name: string) => void
  onDelete: (name: string) => void
}

export function SavedTreesCard({
  savedTrees,
  saveName,
  loading,
  onSaveNameChange,
  onSave,
  onLoad,
  onDelete,
}: SavedTreesCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Saved Trees</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <div className="flex gap-1.5">
          <Input
            value={saveName}
            onChange={e => onSaveNameChange(e.target.value)}
            placeholder="Tree name..."
            className="h-7 text-[11px] flex-1"
            aria-label="Tree name"
          />
          <Button
            size="sm"
            onClick={onSave}
            disabled={!saveName.trim()}
            className="h-7 text-[11px] shrink-0"
          >
            Save Current
          </Button>
        </div>
        {savedTrees.length === 0 ? (
          <p className="text-[10px] text-muted-foreground/60">No saved trees.</p>
        ) : (
          <div className="space-y-0.5">
            {savedTrees.map(t => (
              <div key={t.name} className="flex items-center gap-2 rounded-lg bg-muted/20 px-2.5 py-1.5">
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] font-medium">{t.name}</div>
                  <div className="text-[10px] text-muted-foreground/60">{t.vocab_size} vocab, {t.num_merges} merges</div>
                </div>
                <Button size="sm" variant="ghost" className="h-6 px-1.5 text-[10px]" onClick={() => onLoad(t.name)}>Load</Button>
                <Button size="sm" variant="ghost" className="h-6 px-1.5 text-[10px] text-destructive" onClick={() => onDelete(t.name)}>Delete</Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
