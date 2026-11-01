'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'

interface VisualDatasetCardProps {
  creatingDataset: boolean
  onCreate: (name: string, imageDir: string) => void
}

export default function VisualDatasetCard({ creatingDataset, onCreate }: VisualDatasetCardProps) {
  const [visualDatasetName, setVisualDatasetName] = useState('')
  const [visualImageDir, setVisualImageDir] = useState('')

  const handleCreate = () => {
    if (!visualDatasetName.trim() || !visualImageDir.trim()) return
    onCreate(visualDatasetName.trim(), visualImageDir.trim())
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Image description dataset</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">Create a training dataset from a folder of images. The AI writes descriptions for each image automatically.</p>
        <div className="flex items-center gap-2">
          <Input value={visualDatasetName} onChange={e => setVisualDatasetName(e.target.value)} placeholder="Dataset name" className="h-8 text-xs flex-1" aria-label="Visual dataset name" />
          <Input value={visualImageDir} onChange={e => setVisualImageDir(e.target.value)} placeholder="/path/to/images" className="h-8 text-xs flex-1" aria-label="Image directory for visual dataset" />
        </div>
        <Button size="sm" className="h-8 text-xs" onClick={handleCreate} disabled={creatingDataset || !visualDatasetName.trim() || !visualImageDir.trim()}>
          {creatingDataset ? 'Creating…' : 'Create dataset'}
        </Button>
      </CardContent>
    </Card>
  )
}
