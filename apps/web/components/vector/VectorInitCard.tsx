'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

export interface VectorInitCardProps {
  provider: string
  loading: boolean
  onInit: (provider: string) => void
}

export function VectorInitCard({ provider, loading, onInit }: VectorInitCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2 pt-2.5 px-2.5">
        <CardTitle className="text-[11px] font-medium">Initialize</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 px-2.5 pb-2.5">
        <p className="text-[10px] text-muted-foreground/60">Choose a vector store backend. ChromaDB persists to disk; in-memory is faster but does not survive restarts.</p>
        <div className="flex gap-1">
          <Button
            size="sm"
            variant={provider === 'in_memory' ? 'default' : 'outline'}
            className="h-7 text-[11px]"
            onClick={() => onInit('in_memory')}
            disabled={loading}
          >
            In Memory
          </Button>
          <Button
            size="sm"
            variant={provider === 'chromadb' ? 'default' : 'outline'}
            className="h-7 text-[11px]"
            onClick={() => onInit('chromadb')}
            disabled={loading}
          >
            ChromaDB
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
