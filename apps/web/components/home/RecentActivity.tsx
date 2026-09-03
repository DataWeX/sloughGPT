'use client'

import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Card, CardContent, cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface RecentActivityProps {
  apiStatus: string
  loading: boolean
  modelStatus: { loaded: boolean; model: string | null }
  recentSessions: Array<{ id: string; name: string; updated_at: string; message_count?: number; pinned?: boolean; starred?: boolean }>
  recentJobs: Array<{ id: string; name: string; status: string; created_at?: string }>
  recentDatasets: Array<{ id: string; name: string; updated_at?: string; size?: number; samples?: number }>
}

export function RecentActivity({ apiStatus, loading, modelStatus, recentSessions, recentJobs, recentDatasets }: RecentActivityProps) {
  const router = useRouter()

  if (loading) {
    return (
      <Card>
        <CardContent className="py-3">
          <div className="h-4 w-28 animate-pulse rounded bg-muted mb-2" />
          <div className="space-y-1.5">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-center gap-2 px-1.5 py-1">
                <span className="w-1.5 h-1.5 rounded-full animate-pulse bg-muted shrink-0" />
                <span className="h-3 flex-1 animate-pulse rounded bg-muted" />
                <span className="h-3 w-12 animate-pulse rounded bg-muted shrink-0" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (apiStatus !== 'online' || !modelStatus.loaded) return null

  return (
    <>
      <Card>
        <CardContent className="py-3">
          <p className="text-sm font-medium mb-2">Recent activity</p>
          <div className="space-y-1.5">
            {recentSessions.slice(0, 3).map(s => (
              <button
                 key={s.id}
                 type="button"
                 onClick={() => router.push(`/chat?session=${s.id}`)}
                 className="w-full flex items-center gap-2 text-left hover:bg-muted/30 rounded px-1.5 py-1 transition-colors"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-primary/60 shrink-0" />
                <span className="text-xs truncate flex-1">{s.name}</span>
                {s.starred && <span className="text-xs shrink-0">★</span>}
                {s.pinned && <span className="text-xs text-primary shrink-0">📌</span>}
                <span className="text-xs text-muted-foreground shrink-0">
                  {s.message_count != null && <span>{s.message_count}m · </span>}
                  {timeAgo(s.updated_at)}
                </span>
              </button>
            ))}
            {recentJobs.slice(0, 2).map(j => (
              <div key={j.id} className="flex items-center gap-2 px-1.5 py-1">
                <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', j.status === 'running' ? 'bg-success animate-pulse' : j.status === 'completed' ? 'bg-success' : j.status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/40')} />
                <span className="text-xs truncate flex-1">{j.name || j.id}</span>
                <span className={cn('text-xs px-1.5 py-0.5 rounded font-medium', j.status === 'running' ? 'bg-warning/15 text-warning' : j.status === 'completed' ? 'bg-success/15 text-success' : j.status === 'failed' ? 'bg-destructive/15 text-destructive' : 'bg-muted text-muted-foreground')}>{j.status}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {recentDatasets.length > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-sm font-medium">Recent datasets</p>
              <Link href="/datasets" prefetch={false} className="text-xs text-primary hover:text-primary/80 ml-auto">View all →</Link>
            </div>
            <div className="space-y-1.5">
              {recentDatasets.map(ds => (
                <button
                  key={ds.id}
                  type="button"
                  onClick={() => router.push(`/training?dataset=${encodeURIComponent(ds.id)}`)}
                  className="w-full flex items-center gap-2 text-left hover:bg-muted/30 rounded px-1.5 py-1 transition-colors"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent/60 shrink-0" />
                  <span className="text-xs truncate flex-1">{ds.name}</span>
                  {ds.samples != null && <span className="text-xs text-muted-foreground shrink-0">{ds.samples.toLocaleString()} samples</span>}
                  <span className="text-xs text-primary shrink-0">Train →</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </>
  )
}
