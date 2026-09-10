'use client'

import { Card, CardHeader, CardTitle, CardContent, Badge, cn } from '@sloughgpt/strui'
import { Activity } from 'lucide-react'

interface AgentRun {
  id: string
  status: 'completed' | 'failed' | 'running'
  task: string
  result?: string
  started_at: string
  completed_at?: string
}

interface AgentRunCardProps {
  runs: AgentRun[]
}

function StatusDot({ status }: { status: AgentRun['status'] }) {
  const color = status === 'completed'
    ? 'bg-green-500'
    : status === 'failed'
      ? 'bg-red-500'
      : 'bg-yellow-500 animate-pulse'

  return <span className={cn('inline-block h-2 w-2 rounded-full', color)} />
}

function formatTimestamp(ts: string) {
  return new Date(ts).toLocaleString()
}

export function AgentRunCard({ runs }: AgentRunCardProps) {
  return (
    <Card data-testid="agent-run-card">
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Recent Runs
        </CardTitle>
      </CardHeader>
      <CardContent>
        {runs.length === 0 ? (
          <div className="text-sm text-muted-foreground">No runs yet.</div>
        ) : (
          <div className="space-y-2">
            {runs.map(run => (
              <div
                key={run.id}
                className="flex items-start gap-3 p-2.5 rounded border border-border"
              >
                <StatusDot status={run.status} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium truncate">{run.task}</span>
                    <Badge variant="secondary" className="text-[9px] shrink-0">{run.status}</Badge>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[9px] text-muted-foreground">
                      Started {formatTimestamp(run.started_at)}
                    </span>
                    {run.completed_at && (
                      <span className="text-[9px] text-muted-foreground">
                        Completed {formatTimestamp(run.completed_at)}
                      </span>
                    )}
                  </div>
                  {run.result && (
                    <div className="mt-1 text-[10px] text-muted-foreground line-clamp-2 font-mono">
                      {run.result}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
