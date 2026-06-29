'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function VideoTrainingCard() {
  const addToast = useToastStore(s => s.addToast)
  const [videoDataPath, setVideoDataPath] = useState('')
  const [videoEpochs, setVideoEpochs] = useState(5)
  const [videoBatchSize, setVideoBatchSize] = useState(2)
  const [videoTrainRunning, setVideoTrainRunning] = useState(false)
  const [videoTrainStatus, setVideoTrainStatus] = useState<{
    status: string; current_epoch: number; current_step: number
    total_steps: number; current_loss: number | null; result: any; error: string | null
  } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleStartVideoTrain = async () => {
    if (!videoDataPath) return
    setVideoTrainRunning(true)
    try {
      const result = await visualController.startVideoTrain({
        data_path: videoDataPath,
        epochs: videoEpochs,
        batch_size: videoBatchSize,
      })
      addToast(`Video training started: ${result.job_id}`, 'success')
      pollRef.current = setInterval(async () => {
        try {
          const st = await visualController.getVideoTrainStatus()
          setVideoTrainStatus(st)
          if (st.status === 'completed' || st.status === 'error') {
            if (pollRef.current) clearInterval(pollRef.current)
            setVideoTrainRunning(false)
            if (st.status === 'completed') addToast('Video training complete!', 'success')
            else if (st.error) addToast(`Video training failed: ${st.error}`, 'error')
          }
        } catch { if (pollRef.current) clearInterval(pollRef.current); setVideoTrainRunning(false) }
      }, 3000)
    } catch (err: any) {
      addToast(`Failed to start video training: ${err.message}`, 'error')
      setVideoTrainRunning(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Video Training</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          value={videoDataPath}
          onChange={(e) => setVideoDataPath(e.target.value)}
          placeholder="Video JSONL data path"
          className="text-sm"
        />
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Epochs</label>
            <Input
              type="number"
              value={videoEpochs}
              onChange={(e) => setVideoEpochs(parseInt(e.target.value) || 5)}
              className="text-sm"
              min={1}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Batch Size</label>
            <Input
              type="number"
              value={videoBatchSize}
              onChange={(e) => setVideoBatchSize(parseInt(e.target.value) || 2)}
              className="text-sm"
              min={1}
            />
          </div>
        </div>
        <Button
          size="sm"
          onClick={handleStartVideoTrain}
          disabled={!videoDataPath || videoTrainRunning || videoTrainStatus?.status === 'running'}
        >
          {videoTrainRunning ? 'Starting...' : videoTrainStatus?.status === 'running' ? 'Training...' : 'Start Video Training'}
        </Button>
        {videoTrainStatus?.status === 'running' && (
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">
              Epoch {videoTrainStatus.current_epoch} | Step {videoTrainStatus.current_step}/{videoTrainStatus.total_steps}
              {videoTrainStatus.current_loss != null && ` | loss: ${videoTrainStatus.current_loss.toFixed(4)}`}
            </p>
            <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
              <div className="h-full bg-primary transition-all" style={{ width: `${videoTrainStatus.total_steps > 0 ? (videoTrainStatus.current_step / videoTrainStatus.total_steps) * 100 : 0}%` }} />
            </div>
          </div>
        )}
        {videoTrainStatus?.status === 'completed' && (
          <p className="text-xs text-success">Training complete!</p>
        )}
      </CardContent>
    </Card>
  )
}
