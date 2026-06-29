'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function VideoInferenceCard() {
  const addToast = useToastStore(s => s.addToast)
  const [videoInferPath, setVideoInferPath] = useState('')
  const [videoInferResult, setVideoInferResult] = useState<string | null>(null)
  const [videoInferRunning, setVideoInferRunning] = useState(false)

  const handleVideoInfer = async () => {
    if (!videoInferPath) return
    setVideoInferRunning(true)
    setVideoInferResult(null)
    try {
      const result = await visualController.videoInference({ video_path: videoInferPath })
      setVideoInferResult(result.text)
      addToast(`Video inference: ${result.checkpoint} (${result.elapsed_ms}ms)`, 'info')
    } catch (err: any) {
      addToast(`Video inference failed: ${err.message}`, 'error')
    } finally {
      setVideoInferRunning(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Video Inference</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          value={videoInferPath}
          onChange={(e) => setVideoInferPath(e.target.value)}
          placeholder="Server path to video file"
          className="text-sm"
        />
        <Button size="sm" onClick={handleVideoInfer} disabled={!videoInferPath || videoInferRunning}>
          {videoInferRunning ? 'Generating...' : 'Generate Caption'}
        </Button>
        {videoInferResult && (
          <div className="rounded-lg bg-muted/50 p-3 text-sm font-mono leading-relaxed">
            {videoInferResult}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
