'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@sloughgpt/strui'
import { trainingJobsController, type RecoverableJob } from '@/lib/training-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function RecoveryCard({ addToast }: Props) {
  const [jobs, setJobs] = useState<RecoverableJob[]>([])
  const [loading, setLoading] = useState(true)
  const [recovering, setRecovering] = useState<string | null>(null)
  const [pendingAbandon, setPendingAbandon] = useState<string | null>(null)

  const fetchRecoverable = useCallback(async () => {
    setLoading(true)
    try {
      const result = await trainingJobsController.recoverable()
      setJobs(result ?? [])
    } catch {
      setJobs([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await trainingJobsController.recoverable()
        if (active) setJobs(result ?? [])
      } catch {
        if (active) setJobs([])
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const handleRecover = useCallback(async (id: string) => {
    setRecovering(id)
    try {
      const result = await trainingJobsController.recover(id)
      addToast(`Job recovered: ${result.status}`, 'success')
      void fetchRecoverable()
    } catch {
      addToast('Could not recover job', 'error')
    } finally {
      setRecovering(null)
    }
  }, [addToast, fetchRecoverable])

  const handleAbandon = useCallback(async () => {
    if (!pendingAbandon) return
    const id = pendingAbandon
    setPendingAbandon(null)
    try {
      await trainingJobsController.abandon(id)
      addToast('Job abandoned', 'success')
      void fetchRecoverable()
    } catch {
      addToast('Could not abandon job', 'error')
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

  if (jobs.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Recoverable Jobs ({jobs.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {jobs.map(j => (
            <div key={j.id} className="flex items-center justify-between rounded border p-3 text-sm">
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{j.name || j.id}</p>
                <p className="text-xs text-muted-foreground">
                  Failed {j.failed_at ? new Date(j.failed_at).toLocaleString() : 'recently'}
                </p>
              </div>
              <div className="flex items-center gap-1">
                <Button size="sm" variant="ghost" onClick={() => void handleRecover(j.id)} disabled={recovering === j.id}>
                  {recovering === j.id ? 'Recovering...' : 'Recover'}
                </Button>
                <Button size="sm" variant="ghost" className="text-destructive" onClick={() => setPendingAbandon(j.id)}>
                  Abandon
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>

      <AlertDialog open={pendingAbandon !== null} onOpenChange={(open) => { if (!open) setPendingAbandon(null) }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Abandon this job?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently discard the failed training job. You will not be able to recover it later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction onClick={handleAbandon} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Abandon Job
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
