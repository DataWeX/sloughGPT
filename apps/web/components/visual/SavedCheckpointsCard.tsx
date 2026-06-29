'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface Checkpoint {
  name: string; path: string; size_mb: number; final_loss: number | null
  total_steps: number; vision_encoder?: string; llm?: string
}

interface SavedCheckpointsCardProps {
  loading: boolean
  checkpoints: Checkpoint[]
  onUsePath: (path: string) => void
  onLoad: (name: string) => void
  onDelete: (name: string) => void
}

export default function SavedCheckpointsCard({
  loading, checkpoints, onUsePath, onLoad, onDelete,
}: SavedCheckpointsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Saved Checkpoints</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {loading ? (
          <div className="h-16 animate-pulse rounded bg-muted" />
        ) : checkpoints.length === 0 ? (
          <p className="text-xs text-muted-foreground">No visual checkpoints found on disk.</p>
        ) : (
          checkpoints.map((ck) => (
            <div key={ck.name} className="flex items-center justify-between rounded-lg border p-3">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{ck.name}</p>
                <p className="text-xs text-muted-foreground">
                  {ck.size_mb.toFixed(2)} MB
                  {ck.final_loss != null && ` · loss: ${ck.final_loss.toFixed(4)}`}
                  {ck.total_steps > 0 && ` · ${ck.total_steps} steps`}
                  {ck.llm && ` · ${ck.llm}`}
                </p>
              </div>
              <div className="flex gap-1 ml-3 shrink-0">
                <Button size="sm" variant="ghost" onClick={() => onUsePath(ck.path)}>Use Path</Button>
                <Button size="sm" variant="outline" onClick={() => onLoad(ck.name)}>Load</Button>
                <Button size="sm" variant="destructive" onClick={() => onDelete(ck.name)}>Delete</Button>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
