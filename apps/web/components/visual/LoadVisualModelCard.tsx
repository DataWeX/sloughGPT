'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

interface LoadVisualModelCardProps {
  onLoaded: () => void
}

export default function LoadVisualModelCard({ onLoaded }: LoadVisualModelCardProps) {
  const addToast = useToastStore(s => s.addToast)
  const [loadModelDir, setLoadModelDir] = useState('models/visual-finetuned')
  const [loadingModel, setLoadingModel] = useState(false)

  const handleLoadModel = async () => {
    setLoadingModel(true)
    try {
      const result = await visualController.loadVisualModel(loadModelDir)
      addToast(result.message, 'success')
      onLoaded()
    } catch (err: any) {
      addToast(`Failed: ${err.message}`, 'error')
    } finally {
      setLoadingModel(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Load Visual Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          value={loadModelDir}
          onChange={(e) => setLoadModelDir(e.target.value)}
          placeholder="Model directory path"
          className="text-sm"
        />
        <Button size="sm" onClick={handleLoadModel} disabled={loadingModel}>
          {loadingModel ? 'Loading...' : 'Load Model'}
        </Button>
      </CardContent>
    </Card>
  )
}
