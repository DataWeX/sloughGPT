'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Label, Progress } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function FeedbackTrainCard({ addToast }: Props) {
  const [phase, setPhase] = useState<'idle' | 'starting' | 'training' | 'complete' | 'error'>('idle')
  const [job, setJob] = useState<{ job_id?: string; samples?: number } | null>(null)
  const [progress, setProgress] = useState(0)
  const [loss, setLoss] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadingModel, setLoadingModel] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [epochs, setEpochs] = useState(5)
  const [lr, setLr] = useState(1e-3)
  const [batchSize, setBatchSize] = useState(32)
  const [pendingStop, setPendingStop] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  useEffect(() => () => stopPolling(), [stopPolling])

  const startPolling = useCallback((jobId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const j = await trainingJobsController.get(jobId)
        if (!j) return

        if (j.status === 'running') {
          setProgress(j.progress ?? 0)
          const currentLoss = j.loss ?? j.train_loss
          if (currentLoss != null) setLoss(currentLoss)
          return
        }

        stopPolling()
        if (j.status === 'completed') {
          setPhase('complete')
          setProgress(100)
          addToast('Feedback training complete', 'success')
        } else {
          setPhase('error')
          setError(j.error || 'Could not training')
          addToast('Could not feedback training', 'error')
        }
      } catch {
        // Transient error, keep polling
      }
    }, 3000)
  }, [stopPolling, addToast])

  const handleTrain = useCallback(async () => {
    setPhase('starting')
    setProgress(0)
    setLoss(null)
    setError(null)
    try {
      const resp = await trainingJobsController.trainFromFeedback({ epochs, learning_rate: lr, batch_size: batchSize })
      if (resp.status === 'error') {
        setPhase('error')
        setError(resp.message || 'No feedback data available')
        addToast(resp.message || 'No feedback data available', 'error')
        return
      }
      setJob(resp)
      setPhase('training')
      addToast(`Training started from ${resp.samples ?? 0} feedback pairs`, 'success')
      if (resp.job_id) startPolling(resp.job_id)
    } catch {
      setPhase('error')
      setError('Could not start training')
      addToast('Could not start training from feedback', 'error')
    }
  }, [epochs, lr, batchSize, addToast, startPolling])

  const handleStop = useCallback(async () => {
    stopPolling()
    if (job?.job_id) {
      try { await trainingJobsController.stop(job.job_id) } catch { /* best effort */ }
    }
    setPhase('idle')
    setJob(null)
  }, [job, stopPolling])

  const handleLoad = useCallback(async () => {
    const checkpoint = job?.job_id ? `feedback-trained-${job.job_id}.soul` : null
    if (!checkpoint) return
    setLoadingModel(true)
    try {
      await soulsController.loadCheckpoint(checkpoint)
      addToast(`Loaded: ${checkpoint}`, 'success')
    } catch {
      addToast('Could not load checkpoint', 'error')
    } finally {
      setLoadingModel(false)
    }
  }, [job, addToast])

  const isRunning = phase === 'starting' || phase === 'training'

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train from feedback</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Export your feedback as training data and fine-tune a model.
        </p>

        {isRunning ? (
          <div className="space-y-3" aria-live="polite" aria-atomic="true">
            <Progress value={progress} max={100} />
            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
              <span>Progress {progress}%</span>
              {loss != null && <span>Loss {loss.toFixed(4)}</span>}
              {job?.samples != null && <span>{job.samples} pairs</span>}
            </div>
            <Button variant="destructive" size="sm" onClick={() => setPendingStop(true)}>
              Stop
            </Button>
          </div>
        ) : phase === 'complete' ? (
          <div className="space-y-2">
            <p className="text-sm text-success font-medium">Training complete</p>
            <div className="grid grid-cols-2 gap-x-4 text-xs text-muted-foreground">
              {job?.job_id && (
                <>
                  <span>Job ID</span>
                  <span className="font-mono text-foreground">{job.job_id}</span>
                </>
              )}
              {job?.samples != null && (
                <>
                  <span>Pairs used</span>
                  <span className="font-mono text-foreground">{job.samples}</span>
                </>
              )}
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleLoad} disabled={loadingModel}>
                {loadingModel ? 'Loading...' : 'Load for chat'}
              </Button>
              <Button variant="outline" size="sm" onClick={() => { setPhase('idle'); setJob(null) }}>
                Train again
              </Button>
            </div>
          </div>
        ) : phase === 'error' ? (
          <div className="space-y-2">
            <p className="text-sm text-destructive font-medium">Training failed</p>
            {error && <p className="text-xs text-muted-foreground">{error}</p>}
            <Button variant="outline" size="sm" onClick={() => { setPhase('idle'); setJob(null) }}>
              Dismiss
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Button size="sm" variant="ghost" onClick={() => setShowConfig(!showConfig)}>
              {showConfig ? 'Hide config' : 'Show config'}
            </Button>
            {showConfig && (
              <div className="grid grid-cols-3 gap-2">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="fb-epochs" variant="uppercase">Epochs</Label>
                  <Input id="fb-epochs" type="number" min={1} max={100} value={epochs}
                    onChange={e => setEpochs(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="fb-lr" variant="uppercase">Learning Rate</Label>
                  <Input id="fb-lr" type="text" inputMode="decimal" value={lr}
                    onChange={e => setLr(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="fb-batch" variant="uppercase">Batch Size</Label>
                  <Input id="fb-batch" type="number" min={1} max={256} value={batchSize}
                    onChange={e => setBatchSize(Number(e.target.value))} className="h-8 text-xs font-mono" />
                </div>
              </div>
            )}
            <Button size="sm" onClick={handleTrain}>
              Train from feedback
            </Button>
          </div>
        )}
      </CardContent>

      <AlertDialog open={pendingStop} onOpenChange={setPendingStop}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop training?</AlertDialogTitle>
            <AlertDialogDescription>
              Training will be cancelled and progress will be lost. You can recover the job later from the Settings tab.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep training</AlertDialogCancel>
            <AlertDialogAction onClick={handleStop} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Stop Training
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
