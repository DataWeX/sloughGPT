'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface ExportJob {
  id: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message?: string
  startedAt: number
  completedAt?: number
  error?: string
}

const STORAGE_KEY = 'sloughgpt-export-jobs'

function loadJobs(): ExportJob[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveJobs(jobs: ExportJob[]) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.slice(0, 20))) } catch { /* quota */ }
}

export function startExportJob(type: string): string {
  const id = `job-${Date.now()}`
  const jobs = loadJobs()
  jobs.unshift({
    id,
    type,
    status: 'running',
    progress: 0,
    startedAt: Date.now(),
  })
  saveJobs(jobs)
  return id
}

export function updateExportJob(id: string, updates: Partial<Pick<ExportJob, 'progress' | 'status' | 'message' | 'error'>>) {
  const jobs = loadJobs()
  const idx = jobs.findIndex(j => j.id === id)
  if (idx >= 0) {
    jobs[idx] = { ...jobs[idx], ...updates }
    if (updates.status === 'completed' || updates.status === 'failed') {
      jobs[idx].completedAt = Date.now()
    }
    saveJobs(jobs)
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${s % 60}s`
}

const STATUS_STYLE: Record<string, string> = {
  pending: 'bg-muted text-muted-foreground',
  running: 'bg-primary/15 text-primary',
  completed: 'bg-success/15 text-success',
  failed: 'bg-destructive/15 text-destructive',
}

export function ExportProgressCard() {
  const [jobs, setJobs] = useState<ExportJob[]>([])
  const [expanded, setExpanded] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const refresh = useCallback(() => {
    setJobs(loadJobs())
  }, [])

  useEffect(() => {
    refresh()
    intervalRef.current = setInterval(refresh, 2000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [refresh])

  const clearCompleted = useCallback(() => {
    const active = jobs.filter(j => j.status === 'running' || j.status === 'pending')
    setJobs(active)
    saveJobs(active)
  }, [jobs])

  if (jobs.length === 0) return null

  const running = jobs.filter(j => j.status === 'running')
  const displayJobs = expanded ? jobs : jobs.slice(0, 5)

  return (
    <Card data-testid="export-progress">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Export Progress</CardTitle>
          <div className="flex items-center gap-1">
            {running.length > 0 && (
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/15 text-primary font-medium animate-pulse">
                {running.length} active
              </span>
            )}
            <Button size="sm" variant="ghost" className="h-5 text-[9px]" onClick={clearCompleted}>
              Clear done
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {displayJobs.map(job => (
          <div key={job.id} className="rounded-md border border-border/40 px-2.5 py-2 text-[11px]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', STATUS_STYLE[job.status])}>
                  {job.status}
                </span>
                <span className="font-medium">{job.type}</span>
              </div>
              <span className="text-[9px] text-muted-foreground">
                {formatDuration((job.completedAt ?? Date.now()) - job.startedAt)}
              </span>
            </div>

            {(job.status === 'running' || job.status === 'pending') && (
              <div className="mt-1.5">
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <div className="flex justify-between mt-0.5">
                  <span className="text-[9px] text-muted-foreground">{job.progress}%</span>
                  {job.message && <span className="text-[9px] text-muted-foreground">{job.message}</span>}
                </div>
              </div>
            )}

            {job.status === 'failed' && job.error && (
              <p className="mt-1 text-[9px] text-destructive">{job.error}</p>
            )}
          </div>
        ))}

        {jobs.length > 5 && (
          <Button size="sm" variant="ghost" className="h-5 text-[9px] w-full" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show less' : `Show all ${jobs.length}`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
