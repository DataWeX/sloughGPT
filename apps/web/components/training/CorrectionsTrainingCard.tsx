'use client'

import { useCallback, useEffect, useState } from 'react'
import Link from '@/vite/next-compat/link'
import { Button, StatusDot } from '@sloughgpt/strui'
import { feedbackController } from '@/lib/feedback-controller'

export function CorrectionsTrainingCard() {
  const [count, setCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [starting, setStarting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const stats = await feedbackController.getFeedbackStats()
      setCount(stats.corrected_examples ?? 0)
      setError(null)
    } catch {
      setError('Could not check saved corrections.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const interval = window.setInterval(() => void refresh(), 15_000)
    return () => window.clearInterval(interval)
  }, [refresh])

  const startTraining = async () => {
    setStarting(true)
    setError(null)
    try {
      const result = await feedbackController.trainFromCorrections()
      setJobId(result.job_id)
    } catch {
      setError('Training could not start. Save a correction and try again.')
      void refresh()
    } finally {
      setStarting(false)
    }
  }

  return (
    <section className="space-y-3 rounded-xl border border-border/60 bg-card p-4" aria-labelledby="corrections-training-title">
      <div className="flex items-center gap-2">
        <StatusDot tone={count > 0 ? 'success' : 'muted'} />
        <h2 id="corrections-training-title" className="font-semibold">Learn from your corrections</h2>
      </div>
      <p className="text-sm text-muted-foreground">
        {loading
          ? 'Checking saved corrections…'
          : count > 0
            ? `${count} corrected ${count === 1 ? 'answer is' : 'answers are'} ready to train on.`
            : 'Correct an answer in chat to create a real training example.'}
      </p>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={startTraining} disabled={loading || starting || count === 0}>
          {starting ? 'Starting training…' : 'Build candidate model'}
        </Button>
        {jobId && (
          <Link href={`/training/job/${encodeURIComponent(jobId)}`} className="text-sm underline underline-offset-4">
            Watch training progress
          </Link>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        Training builds a separate candidate checkpoint. It does not replace the model currently answering chats.
      </p>
    </section>
  )
}
