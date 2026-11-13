'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, cn } from '@sloughgpt/strui'

const SOURCE_TYPES = ['file', 'url', 'rss', 'api', 'sse', 'watch', 'generator']
const STORE_TYPES = ['memory', 'file', 'callback', 'chained', 'stats']

interface CollectionCreateCardProps {
  onCreate?: (name: string, sourceType: string, storeType: string) => Promise<void>
}

export function CollectionCreateCard({ onCreate }: CollectionCreateCardProps) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [sourceType, setSourceType] = useState('file')
  const [storeType, setStoreType] = useState('memory')
  const [creating, setCreating] = useState(false)

  const handleCreate = async () => {
    if (!name.trim() || !onCreate) return
    setCreating(true)
    try {
      await onCreate(name.trim(), sourceType, storeType)
      setName('')
      setOpen(false)
    } finally {
      setCreating(false)
    }
  }

  return (
    <Card data-testid="collection-create">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">New Pipeline</CardTitle>
          <Button size="sm" variant="ghost" className="text-[10px]" onClick={() => setOpen(!open)}>
            {open ? 'Cancel' : '+ Create'}
          </Button>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Name</div>
            <Input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="my-pipeline"
              data-testid="pipeline-name"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Source</div>
              <select
                value={sourceType}
                onChange={e => setSourceType(e.target.value)}
                className="w-full text-xs border border-border rounded px-2 py-1.5 bg-background"
                data-testid="pipeline-source"
              >
                {SOURCE_TYPES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Store</div>
              <select
                value={storeType}
                onChange={e => setStoreType(e.target.value)}
                className="w-full text-xs border border-border rounded px-2 py-1.5 bg-background"
                data-testid="pipeline-store"
              >
                {STORE_TYPES.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
          <Button
            size="sm"
            onClick={handleCreate}
            disabled={creating || !name.trim()}
          >
            {creating ? 'Creating...' : 'Create Pipeline'}
          </Button>
        </CardContent>
      )}
    </Card>
  )
}
