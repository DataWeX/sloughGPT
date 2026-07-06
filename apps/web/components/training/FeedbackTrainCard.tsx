'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingController } from '@/lib/controllers'
import { feedbackController } from '@/lib/feedback-controller'
import { modelController } from '@/lib/controllers'

export function FeedbackTrainCard({ onComplete }: { onComplete?: () => void }) {
  const addToast = useToastStore(s => s.addToast)
  const [feedbackTraining, setFeedbackTraining] = useState(false)
  const [feedbackComplete, setFeedbackComplete] = useState(false)
  const [feedbackResult, setFeedbackResult] = useState<{ model_path?: string; final_loss?: number; samples?: number } | null>(null)
  const [feedbackError, setFeedbackError] = useState('')
  const [feedbackStats, setFeedbackStats] = useState<{ total: number; thumbs_up: number; thumbs_down: number } | null>(null)
  const fbIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    feedbackController.getFeedbackStats().then(s => {
      const db = s?.db_stats
      if (db) setFeedbackStats({ total: db.feedback_total, thumbs_up: db.thumbs_up, thumbs_down: db.thumbs_down })
    }).catch(() => {})
  }, [])

  useEffect(() => () => {
    if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current) }
  }, [])

  const startFeedbackTrain = useCallback(async () => {
    setFeedbackTraining(true)
    setFeedbackComplete(false)
    setFeedbackError('')
    setFeedbackResult(null)
    try {
      const res = await trainingController.trainFromFeedback({ epochs: 3, batch_size: 16, use_lora: true })
      if (res.status === 'no_data') {
        setFeedbackTraining(false)
        addToast('No feedback data available for training', 'error')
        return
      }
      const jid = res.job_id
      if (!jid) {
        setFeedbackTraining(false)
        addToast('Failed to start feedback training', 'error')
        return
      }

      // Poll job status
      let retries = 0
      const poll = async () => {
        try {
          const jobs = await trainingController.list()
          const job = jobs.find((j: import('@/lib/training-controller').TrainingJob) => j.id === jid)
          if (!job) return
          if (job.status === 'completed') {
            setFeedbackTraining(false)
            setFeedbackComplete(true)
            setFeedbackResult({
              model_path: job.checkpoint,
              final_loss: job.loss ?? job.train_loss,
              samples: (job as any).samples_used,
            })
            onComplete?.()
            addToast('Feedback training complete!', 'success')
            if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current); fbIntervalRef.current = null }
          } else if (job.status === 'failed') {
            setFeedbackTraining(false)
            setFeedbackError(job.message || 'Training failed')
            addToast(job.message || 'Feedback training failed', 'error')
            if (fbIntervalRef.current) { clearInterval(fbIntervalRef.current); fbIntervalRef.current = null }
          }
        } catch { retries++ }
      }
      fbIntervalRef.current = setInterval(poll, 3000)
    } catch {
      setFeedbackTraining(false)
      addToast('Failed to start feedback training', 'error')
    }
  }, [addToast, onComplete])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Train from Feedback</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {feedbackStats && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>{feedbackStats.total} feedback entries</span>
            <span className="text-success">{feedbackStats.thumbs_up} thumbs up</span>
            <span className="text-destructive">{feedbackStats.thumbs_down} thumbs down</span>
          </div>
        )}
        {!feedbackStats && !feedbackTraining && !feedbackComplete && !feedbackError && (
          <p className="text-xs text-muted-foreground">Loading feedback stats...</p>
        )}

        {feedbackError && (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 space-y-2">
            <p className="text-sm font-medium text-destructive">Training failed</p>
            <p className="text-xs text-muted-foreground">{feedbackError}</p>
            <Button size="sm" variant="outline" onClick={() => setFeedbackError('')}>Dismiss</Button>
          </div>
        )}

        {feedbackComplete && feedbackResult && (
          <div className="rounded-lg border border-success/20 bg-success/5 p-3 space-y-2">
            <p className="text-sm font-medium text-success">Feedback training complete!</p>
            {feedbackResult.final_loss != null && (
              <p className="text-xs text-muted-foreground">Final loss: {feedbackResult.final_loss.toFixed(4)}</p>
            )}
            {feedbackResult.samples != null && (
              <p className="text-xs text-muted-foreground">Samples used: {feedbackResult.samples}</p>
            )}
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={async () => {
                if (feedbackResult.model_path) {
                  try {
                    await modelController.loadModelPath(feedbackResult.model_path)
                    addToast('Model loaded for chat', 'success')
                  } catch { addToast('Failed to load model', 'error') }
                }
              }}>Load for chat</Button>
              <Button size="sm" variant="ghost" onClick={() => {
                setFeedbackComplete(false); setFeedbackResult(null)
              }}>Dismiss</Button>
            </div>
          </div>
        )}

        {feedbackTraining && (
          <div className="space-y-2" role="status" aria-live="polite">
            <p className="text-sm text-muted-foreground">Training from feedback data...</p>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary animate-pulse rounded-full" style={{ width: '50%' }} />
            </div>
          </div>
        )}

        {!feedbackTraining && !feedbackComplete && !feedbackError && (
          <Button size="sm" disabled={!feedbackStats || feedbackStats.total === 0} onClick={startFeedbackTrain}>
            {!feedbackStats ? 'Loading...' : feedbackStats.total === 0 ? 'No feedback data' : `Train on ${feedbackStats.total} entries`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
