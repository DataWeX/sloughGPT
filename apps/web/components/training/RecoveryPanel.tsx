'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { FoldSection } from '@/components/strui'
import { trainingJobsController, type TrainingJob } from '@/lib/training-controller'
import { devDebug } from '@/lib/dev-log'
import { useToastStore } from '@/lib/toast-store'

interface RecoveryPanelProps {
  jobs: TrainingJob[]
  fetchJobs: () => void
}

export function RecoveryPanel({ jobs, fetchJobs }: RecoveryPanelProps) {
  const addToast = useToastStore(s => s.addToast)
  const [expanded, setExpanded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<{
    pending: number
    running: number
    completed: number
    failed: number
    total: number
    crashed_jobs: number
    recoverable_jobs: number
  } | null>(null)
  const [recoverableJobs, setRecoverableJobs] = useState<Array<{
    id: string
    name: string
    status: string
    progress: number
    config: Record<string, unknown>
    checkpoint_path?: string
  }>>([])

  const checkRecovery = useCallback(async () => {
    try {
      const data = await trainingJobsController.getRecoveryStats() as Record<string, number>
      setStats(data as any)
      const recoverable = await trainingJobsController.recoverable()
      setRecoverableJobs(recoverable.map(j => ({ ...j, status: 'failed', progress: 0, config: {}, checkpoint_path: undefined })))
    } catch (error) {
      devDebug('Failed to check recovery:', error)
    }
  }, [])

  useEffect(() => {
    void checkRecovery()
    const interval = setInterval(() => void checkRecovery(), 30000)
    return () => clearInterval(interval)
  }, [checkRecovery])

  const handleRecover = async (jobId: string) => {
    if (!confirm('Recover this job? It will restart from the last checkpoint.')) return
    setLoading(true)
    try {
      const result = await trainingJobsController.recover(jobId)
      addToast('Job recovery ' + (result.status === 'recovered' ? 'started' : 'triggered'), result.status === 'recovered' ? 'success' : 'info')
      void fetchJobs()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Recovery failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleAbandon = async (jobId: string) => {
    if (!confirm('Abandon this job permanently? This cannot be undone.')) return
    setLoading(true)
    try {
      await trainingJobsController.abandon(jobId)
      addToast('Job abandoned', 'success')
      void fetchJobs()
      void checkRecovery()
    } catch (error) {
      addToast(error instanceof Error ? error.message : 'Abandon failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const crashedCount = stats?.crashed_jobs ?? 0

  if (crashedCount === 0) return null

  return (
    <FoldSection heading="Job Recovery">
      <div className="flex items-center justify-between p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
        <div>
          <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
            {crashedCount > 0 ? `${crashedCount} job(s) may have crashed` : 'Interrupted jobs detected'}
          </p>
          <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
            Server may have stopped unexpectedly.
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Hide' : 'Show'}
        </Button>
      </div>

      {expanded && (
        <div className="mt-3 space-y-2">
          {recoverableJobs.length > 0 ? (
            recoverableJobs.map(job => (
              <div key={job.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-md">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="font-medium text-sm truncate">{job.name || job.id}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span>Progress: {job.progress}%</span>
                    {job.checkpoint_path && (
                      <span className="truncate">Checkpoint: {job.checkpoint_path.split('/').pop()}</span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 ml-2">
                  <Button
                    size="sm"
                    variant="default"
                    onClick={() => void handleRecover(job.id)}
                    disabled={loading}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    Recover
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => void handleAbandon(job.id)}
                    className="text-destructive/60 hover:text-destructive"
                  >
                    Abandon
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              No recoverable jobs found.
            </p>
          )}
        </div>
      )}
    </FoldSection>
  )
}
