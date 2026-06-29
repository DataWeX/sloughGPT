'use client'

import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui'
import { trainingController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

export function JobHistoryCard({
  allJobs,
  checkpoints,
  loadingTimedOut,
  onRetry,
}: {
  allJobs: any[]
  checkpoints: UseTrainingCheckpointsReturn
  loadingTimedOut: boolean
  onRetry: () => void
}) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const hasJobs = allJobs.length > 0 || checkpoints.loadingJobs
  if (!hasJobs) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Job history</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {checkpoints.loadingJobs ? (
          loadingTimedOut ? (
            <div className="px-4 py-6 text-center space-y-2">
              <p className="text-sm text-muted-foreground">Taking longer than expected</p>
              <Button size="sm" variant="ghost" onClick={onRetry}>
                Retry
              </Button>
            </div>
          ) : (
            <div className="divide-y divide-border/50">
              {[1,2,3].map(i => (
                <div key={i} className="flex items-center justify-between px-4 py-3">
                  <div className="space-y-1.5 flex-1">
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-3 w-32" />
                  </div>
                  <Skeleton className="h-5 w-12 rounded-full" />
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="divide-y divide-border/50">
            {allJobs.slice().reverse().map((job) => {
              const relativeTime = (() => {
                if (!job.created_at) return ''
                const diff = Date.now() - new Date(job.created_at).getTime()
                const mins = Math.floor(diff / 60000)
                if (mins < 1) return 'just now'
                if (mins < 60) return `${mins}m ago`
                const hrs = Math.floor(mins / 60)
                if (hrs < 24) return `${hrs}h ago`
                return `${Math.floor(hrs / 24)}d ago`
              })()
              return (
              <div key={job.id} role="button" tabIndex={0} className="flex items-center justify-between px-4 py-3 text-sm cursor-pointer hover:bg-muted/20 transition-colors" onClick={() => router.push(`/training/job/${job.id}`)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); router.push(`/training/job/${job.id}`) } }} aria-label={`View job ${job.name || job.id}`}>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{job.name || job.id}</p>
                  {job.status_message ? (
                    <p className="text-xs text-muted-foreground mt-0.5">{job.status_message}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground mt-0.5">{job.status} &middot; {relativeTime}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-3" onClick={e => e.stopPropagation()}>
                  {job.status === 'running' && (
                    <>
                      <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-success" /></span>
                      <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={async () => {
                        try { await trainingController.stop(job.id); addToast('Stopped', 'info'); void checkpoints.fetchJobs() }
                        catch { addToast('Failed to stop job', 'error') }
                      }}>
                        Stop
                      </Button>
                    </>
                  )}
                  {job.status === 'completed' && (
                    <>
                      {job.checkpoint && (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                          try { await checkpoints.handleLoadCheckpoint(job.checkpoint!, addToast) }
                          catch { addToast('Failed to load trained version', 'error') }
                        }}>
                          Use
                        </Button>
                      )}
                      <span className="text-xs text-success shrink-0">Done</span>
                    </>
                  )}
                  {job.status === 'failed' && (
                    <>
                      <span className="text-xs text-destructive shrink-0">Failed</span>
                      <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => router.push('/training')}>
                        Retry
                      </Button>
                    </>
                  )}
                  {['stopping', 'stopped'].includes(job.status) && (
                    <span className="text-xs text-muted-foreground shrink-0">Stopped</span>
                  )}
                  <Button size="sm" variant="ghost" className="h-6 text-xs text-muted-foreground hover:text-destructive" onClick={async () => {
                    if (!confirm(`Delete job "${job.name || job.id}"?`)) return
                    try {
                      await trainingController.delete(job.id)
                      addToast('Job deleted', 'info')
                      void checkpoints.fetchJobs()
                    } catch { addToast('Failed to delete job', 'error') }
                  }}>
                    Delete
                  </Button>
                </div>
              </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
