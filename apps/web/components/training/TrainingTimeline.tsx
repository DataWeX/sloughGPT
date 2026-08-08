'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TimelineEvent {
  id: string
  type: 'checkpoint' | 'training' | 'load'
  name: string
  timestamp?: string
  loss?: number | null
  detail?: string
  status?: string
}

interface TrainingTimelineProps {
  checkpoints: Checkpoint[]
  maxEvents?: number
}

function formatTime(ts?: string): string {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return ''
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffM = Math.floor(diffMs / 60000)
    const diffH = Math.floor(diffM / 60)
    const diffD = Math.floor(diffH / 24)
    if (diffD > 0) return `${diffD}d ago`
    if (diffH > 0) return `${diffH}h ago`
    if (diffM > 0) return `${diffM}m ago`
    return 'just now'
  } catch {
    return ''
  }
}

function lossIndicator(loss?: number | null): { color: string; label: string } {
  if (loss == null) return { color: 'bg-muted-foreground/30', label: '' }
  if (loss < 1) return { color: 'bg-emerald-500', label: loss.toFixed(3) }
  if (loss < 2) return { color: 'bg-green-500', label: loss.toFixed(3) }
  if (loss < 3) return { color: 'bg-yellow-500', label: loss.toFixed(3) }
  return { color: 'bg-orange-500', label: loss.toFixed(3) }
}

function buildTimeline(checkpoints: Checkpoint[]): TimelineEvent[] {
  const events: TimelineEvent[] = []

  for (const c of checkpoints) {
    events.push({
      id: `cp-${c.name}`,
      type: 'checkpoint',
      name: c.name,
      timestamp: c.born_at,
      loss: c.loss,
      detail: [c.model_type, c.training_dataset].filter(Boolean).join(' · '),
      status: c.verdict,
    })

    if (c.is_loaded) {
      events.push({
        id: `load-${c.name}`,
        type: 'load',
        name: c.name,
        detail: 'Loaded into model',
      })
    }
  }

  events.sort((a, b) => {
    if (!a.timestamp && !b.timestamp) return 0
    if (!a.timestamp) return 1
    if (!b.timestamp) return -1
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  })

  return events
}

export function TrainingTimeline({ checkpoints, maxEvents = 20 }: TrainingTimelineProps) {
  const events = buildTimeline(checkpoints).slice(0, maxEvents)

  if (events.length === 0) return null

  return (
    <Card data-testid="training-timeline">
      <CardHeader className="py-3">
        <CardTitle className="text-base">Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative ml-3 border-l-2 border-muted pl-6 space-y-4">
          {events.map((event, i) => {
            const li = lossIndicator(event.loss)
            return (
              <div key={event.id} className="relative" data-testid="timeline-event">
                <div
                  className={`absolute -left-[31px] top-1 h-3 w-3 rounded-full border-2 border-background ${
                    event.type === 'load' ? 'bg-primary' : li.color
                  }`}
                />
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium truncate">{event.name}</span>
                    {event.type === 'load' && (
                      <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5 bg-primary/10 text-primary border-primary/20">
                        loaded
                      </Badge>
                    )}
                    {event.status && event.status !== 'good' && (
                      <Badge variant="outline" className="text-[9px] px-1 py-0 h-3.5">
                        {event.status}
                      </Badge>
                    )}
                  </div>
                  {event.detail && (
                    <p className="text-[10px] text-muted-foreground">{event.detail}</p>
                  )}
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground/50">
                    {event.timestamp && <span>{formatTime(event.timestamp)}</span>}
                    {event.loss != null && <span className="font-mono">{li.label}</span>}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
