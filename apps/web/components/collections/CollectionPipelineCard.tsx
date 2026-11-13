'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface Pipeline {
  id: string
  name: string
  source_type: string
  store_type: string
  records_count?: number
  last_run?: string
}

interface CollectionPipelineCardProps {
  pipelines: Pipeline[]
  runningId?: string | null
  onRun?: (id: string) => void
  onDelete?: (id: string) => void
}

const SOURCE_COLORS: Record<string, string> = {
  file: 'bg-primary/15 text-primary',
  url: 'bg-accent/15 text-accent',
  rss: 'bg-warning/15 text-warning',
  api: 'bg-success/15 text-success',
  sse: 'bg-destructive/15 text-destructive',
  watch: 'bg-muted text-muted-foreground',
  generator: 'bg-primary/15 text-primary',
}

export function CollectionPipelineCard({ pipelines, runningId, onRun, onDelete }: CollectionPipelineCardProps) {
  if (pipelines.length === 0) return null

  return (
    <Card data-testid="collection-pipeline">
      <CardHeader>
        <CardTitle className="text-base">
          Pipelines
          <span className="text-muted-foreground font-normal ml-2">({pipelines.length})</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {pipelines.map(p => (
            <div key={p.id} className="flex items-center justify-between p-2.5 rounded-lg border border-border hover:border-primary/30 transition-colors">
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium">{p.name}</div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className={cn('text-[9px] px-1.5 py-0.5 rounded font-medium', SOURCE_COLORS[p.source_type] ?? 'bg-muted text-muted-foreground')}>
                    {p.source_type}
                  </span>
                  <span className="text-[9px] text-muted-foreground">→ {p.store_type}</span>
                  {p.records_count != null && (
                    <span className="text-[9px] text-muted-foreground">{p.records_count} records</span>
                  )}
                  {p.last_run && (
                    <span className="text-[9px] text-muted-foreground">Last: {timeAgo(new Date(p.last_run).getTime())}</span>
                  )}
                </div>
              </div>
              <div className="flex gap-0.5">
                {onRun && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={() => onRun(p.id)}
                    disabled={runningId === p.id}
                  >
                    {runningId === p.id ? '...' : 'Run'}
                  </Button>
                )}
                {onDelete && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px] text-destructive"
                    onClick={() => onDelete(p.id)}
                  >
                    Del
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
