'use client'

import { useState, useEffect, useRef } from 'react'
import type { ImportResponse } from '@/lib/dataset-controller'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'

type ImportSource = 'github' | 'huggingface' | 'url' | 'local'

interface ImportJob {
  id: string
  source: ImportSource
  name: string
  status: 'pending' | 'importing' | 'completed' | 'failed'
  progress: number
  message: string
  result?: ImportResponse
  error?: string
}

interface ImportProgressModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  jobs: ImportJob[]
  onAllComplete: () => void
}

const POLL_INTERVAL = 1000

export function ImportProgressModal({
  open,
  onOpenChange,
  jobs,
  onAllComplete,
}: ImportProgressModalProps) {
  const [completedCount, setCompletedCount] = useState(0)
  const [failedCount, setFailedCount] = useState(0)

  const totalJobs = jobs.length
  const completedJobs = jobs.filter((j) => j.status === 'completed').length
  const failedJobs = jobs.filter((j) => j.status === 'failed').length
  const pendingJobs = jobs.filter((j) => j.status === 'pending' || j.status === 'importing').length

  useEffect(() => {
    if (open && pendingJobs === 0 && jobs.length > 0) {
      const timer = setTimeout(() => {
        onAllComplete()
        onOpenChange(false)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [open, pendingJobs, jobs.length, onAllComplete, onOpenChange])

  const overallProgress = totalJobs > 0 ? ((completedJobs + failedJobs) / totalJobs) * 100 : 0

  const statusIcon = (status: ImportJob['status']) => {
    switch (status) {
      case 'completed':
        return '✅'
      case 'failed':
        return '❌'
      case 'importing':
        return '⏳'
      case 'pending':
        return '⏸️'
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Importing Datasets</DialogTitle>
          <DialogDescription>
            {pendingJobs > 0
              ? `Importing ${pendingJobs} dataset${pendingJobs !== 1 ? 's' : ''}...`
              : 'Import complete!'}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4" role="region" aria-label="Import progress">
          <div>
            <div className="mb-2 flex justify-between text-sm">
              <span id="overall-progress-label">Overall Progress</span>
              <span aria-live="polite">{Math.round(overallProgress)}%</span>
            </div>
            <Progress 
              value={overallProgress} 
              className="h-2" 
              aria-labelledby="overall-progress-label"
              aria-valuenow={Math.round(overallProgress)}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>

          <div className="space-y-2 max-h-60 overflow-y-auto" role="list" aria-label="Import jobs">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="rounded-md border p-3"
                role="listitem"
              >
                <div className="flex items-center gap-2">
                  <span aria-hidden="true">{statusIcon(job.status)}</span>
                  <span className="font-medium" aria-live="polite">{job.name}</span>
                  <span className="text-xs text-muted-foreground">({job.source})</span>
                </div>
                <div className="mt-1 text-sm text-muted-foreground" id={`job-${job.id}-status`}>
                  {job.status === 'importing' && (
                    <div className="flex items-center gap-2">
                      <Progress 
                        value={job.progress} 
                        className="h-1 flex-1" 
                        aria-labelledby={`job-${job.id}-label`}
                        aria-valuenow={job.progress}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      />
                      <span aria-live="polite">{job.progress}%</span>
                    </div>
                  )}
                  <span id={`job-${job.id}-label`}>{job.message}</span>
                  {job.error && (
                    <span className="text-destructive" role="alert">{job.error}</span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-between text-sm" role="status" aria-live="polite">
            <span className="text-success">✅ {completedJobs} completed</span>
            <span className="text-destructive">❌ {failedJobs} failed</span>
            <span className="text-muted-foreground">⏳ {pendingJobs} remaining</span>
          </div>
        </div>

        <div className="flex justify-end gap-2">
          {pendingJobs === 0 && (
            <Button onClick={() => { onAllComplete(); onOpenChange(false) }}>
              Done
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
)
}

export function useImportProgress(
  onComplete: () => void
) {
  const [jobs, setJobs] = useState<ImportJob[]>([])
  const [modalOpen, setModalOpen] = useState(false)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const addJob = (source: ImportSource, name: string) => {
    const job: ImportJob = {
      id: `${Date.now()}-${Math.random()}`,
      source,
      name,
      status: 'importing',
      progress: 0,
      message: 'Starting import...',
    }
    setJobs((prev) => [...prev, job])
    setModalOpen(true)
    return job.id
  }

  const updateJob = (jobId: string, updates: Partial<ImportJob>) => {
    setJobs((prev) =>
      prev.map((j) => (j.id === jobId ? { ...j, ...updates } : j))
    )
  }

  const completeJob = (jobId: string, result: ImportResponse) => {
    setJobs((prev) =>
      prev.map((j) =>
        j.id === jobId
          ? { ...j, status: 'completed' as const, progress: 100, message: result.message, result }
          : j
      )
    )
  }

  const failJob = (jobId: string, error: string) => {
    setJobs((prev) =>
      prev.map((j) =>
        j.id === jobId
          ? { ...j, status: 'failed' as const, error }
          : j
      )
    )
  }

  const clearJobs = () => {
    setJobs([])
  }

  useEffect(() => {
    const id = pollIntervalRef.current
    return () => {
      if (id) {
        clearInterval(id)
      }
    }
  }, [])

  return {
    jobs,
    modalOpen,
    setModalOpen,
    addJob,
    updateJob,
    completeJob,
    failJob,
    clearJobs,
    ImportProgressModal: () => (
      <ImportProgressModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        jobs={jobs}
        onAllComplete={() => {
          onComplete()
          clearJobs()
        }}
      />
    ),
  }
}
