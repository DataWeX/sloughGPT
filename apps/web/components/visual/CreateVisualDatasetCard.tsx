'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

interface CreateVisualDatasetCardProps {
  onCreated: () => void
}

export default function CreateVisualDatasetCard({ onCreated }: CreateVisualDatasetCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [datasetName, setDatasetName] = useState('')
  const [datasetDir, setDatasetDir] = useState('')
  const [creatingDataset, setCreatingDataset] = useState(false)

  const handleCreateDataset = async () => {
    if (!datasetName || !datasetDir) return
    setCreatingDataset(true)
    try {
      const result = await visualController.createVisualDataset(datasetName, datasetDir)
      addToast(`Dataset created: ${result.entries} entries from ${datasetDir}`, 'success')
      setDatasetName('')
      setDatasetDir('')
      onCreated()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    } finally {
      setCreatingDataset(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Create Visual Dataset</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          value={datasetName}
          onChange={(e) => setDatasetName(e.target.value)}
          placeholder="Dataset name"
          className="text-sm"
        />
        <Input
          value={datasetDir}
          onChange={(e) => setDatasetDir(e.target.value)}
          placeholder="Path to image directory on server"
          className="text-sm"
        />
        <Button
          size="sm"
          onClick={handleCreateDataset}
          disabled={!datasetName || !datasetDir || creatingDataset}
        >
          {creatingDataset ? 'Creating...' : 'Create Dataset'}
        </Button>
      </CardContent>
    </Card>
  )
}
