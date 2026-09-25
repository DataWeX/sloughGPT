'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { formatDateTime } from '@/lib/time-format'
import { Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@sloughgpt/strui'
import { trainingJobsController, type RecoverableJob } from '@/lib/training-controller'
import { formatToastError } from '@/lib/error-utils'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
  /** Called after a successful recover so the page can refresh jobs + start live polling. */
  onRecovered?: (result: {
    status: string
    original_job_id?: string
    recovery_job_id?: string
    checkpoint_path?: string
    message?: string
  }) => void
}

export function RecoveryCard({ addToast, onRecovered }: Props) {
  const [jobs, setJobs] = useState<RecoverableJob[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [recovering, setRecovering] = useState<string | null>(null)
  const [pendingAbandon, setPendingAbandon] = useState<string | null>(null)

  const activeRef = useRef(true)

  const fetchRecoverable = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await trainingJobsController.recoverable()
      if (activeRef.current) setJobs(result ?? [])
    } catch {
      if (activeRef.current) {
        setJobs([])
        setError('Could not load recoverable jobs')
      }
    } finally {
      if (activeRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    activeRef.current = true
    void fetchRecoverable()
    return () => {
      activeRef.current = false
    }
  }, [fetchRecoverable])

  const handleRecover = useCallback(
    async (id: string) => {
      setRecovering(id)
      try {
        const result = await trainingJobsController.recover(id)
        addToast(result.message || `Job recovered: ${result.status}`, 'success')
        void fetchRecoverable()
        onRecovered?.(result)
      } catch (e) {
        addToast(formatToastError(e, 'Could not recover job'), 'error')
      } finally {
        setRecovering(null)
      }
    },
    [addToast, fetchRecoverable, onRecovered],
  )

  const handleAbandon = useCallback(async () => {
    if (!pendingAbandon) return
    const id = pendingAbandon
    setPendingAbandon(null)
    try {
      await trainingJobsController.abandon(id)
      addToast('Job abandoned', 'success')
      void fetchRecoverable()
    } catch (e) {
      addToast(formatToastError(e, 'Could not abandon job'), 'error')
    }
  }, [pendingAbandon, addToast, fetchRecoverable])

  if (loading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Recoverable Jobs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive/30">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Recoverable Jobs</CardTitle>
        </CardHeader>
        <CardContent>
          <StatusBanner
            variant="error"
            message={error}
            dismissible={false}
            onRetry={() => void fetchRecoverable()}
          />
        </CardContent>
      </Card>
    )
  }

  if (jobs.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Recoverable Jobs ({jobs.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {jobs.map((j) => (
            <div
              key={j.id}
              className="flex items-center justify-between rounded-lg border border-border/40 p-2.5 hover:bg-muted/20 transition-colors"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium text-xs">{j.name || j.id}</p>
                <p className="text-[10px] text-muted-foreground/60">
                  Interrupted{' '}
                  {j.updated_at || j.failed_at
                    ? formatDateTime((j.updated_at || j.failed_at) as string)
                    : 'recently'}
                  {j.progress != null ? ` · ${j.progress}%` : ''}
                </p>
              </div>
              <div className="flex items-center gap-0.5">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-[10px]"
                  onClick={() => void handleRecover(j.id)}
                  disabled={recovering === j.id}
                >
                  {recovering === j.id ? 'Recovering...' : 'Recover'}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-[10px] text-destructive"
                  onClick={() => setPendingAbandon(j.id)}
                >
                  Abandon
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>

      <AlertDialog
        open={pendingAbandon !== null}
        onOpenChange={(open) => {
          if (!open) setPendingAbandon(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Abandon this job?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently discard the failed training job. You will not be able to recover
              it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleAbandon}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Abandon Job
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
