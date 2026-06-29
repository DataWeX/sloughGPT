'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { visualController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'

export default function TrainVisualAICard() {
  const addToast = useToastStore(s => s.addToast)
  const [trainingDataPath, setTrainingDataPath] = useState('')
  const [stage1Epochs, setStage1Epochs] = useState(1)
  const [stage2Epochs, setStage2Epochs] = useState(2)
  const [startingTrain, setStartingTrain] = useState(false)
  const [trainStatus, setTrainStatus] = useState<any>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleStartTrain = async () => {
    if (!trainingDataPath) return
    setStartingTrain(true)
    try {
      const result = await visualController.startVisualTrain({
        data_path: trainingDataPath,
        epochs: stage1Epochs,
      })
      addToast(`Training started: ${result.job_id}`, 'success')
      pollRef.current = setInterval(async () => {
        try {
          const st = await visualController.getVisualTrainStatus()
          setTrainStatus(st)
          if (st.status === 'completed' || st.status === 'error') {
            if (pollRef.current) clearInterval(pollRef.current)
            setStartingTrain(false)
            if (st.status === 'completed') addToast('Training complete!', 'success')
            else if (st.error) addToast(`Training failed: ${st.error}`, 'error')
          }
        } catch { if (pollRef.current) clearInterval(pollRef.current); setStartingTrain(false) }
      }, 3000)
    } catch (err: any) {
      addToast(`Failed to start: ${err.message}`, 'error')
      setStartingTrain(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Train Visual AI</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          value={trainingDataPath}
          onChange={(e) => setTrainingDataPath(e.target.value)}
          placeholder="Training data path (JSONL)"
          className="text-sm"
        />
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground">Stage 1 Epochs</label>
            <Input
              type="number"
              value={stage1Epochs}
              onChange={(e) => setStage1Epochs(parseInt(e.target.value) || 1)}
              className="text-sm"
              min={1}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Stage 2 Epochs</label>
            <Input
              type="number"
              value={stage2Epochs}
              onChange={(e) => setStage2Epochs(parseInt(e.target.value) || 2)}
              className="text-sm"
              min={1}
            />
          </div>
        </div>
        <Button
          size="sm"
          onClick={handleStartTrain}
          disabled={!trainingDataPath || startingTrain || trainStatus?.status === 'running'}
        >
          {startingTrain ? 'Starting...' : trainStatus?.status === 'running' ? 'Training...' : 'Start Training'}
        </Button>
        {trainStatus?.status === 'running' && (
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">
              Stage: {trainStatus.current_stage} | Step {trainStatus.current_step}/{trainStatus.total_steps}
            </p>
            <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
              <div className="h-full bg-primary transition-all" style={{ width: `${(trainStatus.progress || 0) * 100}%` }} />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
