'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'
import { timeAgo } from '@/lib/time-ago'

interface AuthEvent {
  id: string
  type: 'login' | 'logout' | 'register' | 'token_refresh' | 'failed_login'
  timestamp: number
  details?: string
}

const STORAGE_KEY = 'sloughgpt-auth-activity'

function loadActivity(): AuthEvent[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveActivity(events: AuthEvent[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(events))
}

const EVENT_ICONS: Record<string, { icon: string; color: string }> = {
  login: { icon: '→', color: 'text-success' },
  logout: { icon: '←', color: 'text-muted-foreground' },
  register: { icon: '+', color: 'text-primary' },
  token_refresh: { icon: '↻', color: 'text-warning' },
  failed_login: { icon: '✗', color: 'text-destructive' },
}

export function recordAuthEvent(type: AuthEvent['type'], details?: string) {
  const events = loadActivity()
  const entry: AuthEvent = {
    id: `auth-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type,
    timestamp: Date.now(),
    details,
  }
  const updated = [entry, ...events].slice(0, 100)
  saveActivity(updated)
}

export function AuthActivityCard() {
  const [events, setEvents] = useState<AuthEvent[]>(() => loadActivity())
  const [filter, setFilter] = useState<string | null>(null)

  const filtered = filter ? events.filter(e => e.type === filter) : events

  const typeCounts = events.reduce((acc, e) => {
    acc[e.type] = (acc[e.type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const handleClear = () => {
    setEvents([])
    saveActivity([])
  }

  return (
    <Card data-testid="auth-activity">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            Activity
            <span className="text-muted-foreground font-normal ml-2">({events.length})</span>
          </CardTitle>
          {events.length > 0 && (
            <Button size="sm" variant="ghost" className="text-destructive text-[10px]" onClick={handleClear}>
              Clear
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {events.length === 0 ? (
          <div className="text-sm text-muted-foreground">No activity recorded.</div>
        ) : (
          <>
            <div className="flex gap-1 mb-3 flex-wrap">
              <Button
                size="sm"
                variant={filter === null ? 'default' : 'ghost'}
                className="text-[9px]"
                onClick={() => setFilter(null)}
              >
                All ({events.length})
              </Button>
              {Object.entries(typeCounts).map(([type, count]) => (
                <Button
                  key={type}
                  size="sm"
                  variant={filter === type ? 'default' : 'ghost'}
                  className="text-[9px]"
                  onClick={() => setFilter(type)}
                >
                  {type} ({count})
                </Button>
              ))}
            </div>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {filtered.map(evt => {
                const meta = EVENT_ICONS[evt.type] ?? { icon: '•', color: 'text-muted-foreground' }
                return (
                  <div key={evt.id} className="flex items-center gap-2 p-1.5 rounded hover:bg-muted/50">
                    <span className={cn('text-xs font-mono w-4 text-center', meta.color)}>{meta.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium capitalize">{evt.type.replace('_', ' ')}</div>
                      {evt.details && (
                        <div className="text-[10px] text-muted-foreground truncate">{evt.details}</div>
                      )}
                    </div>
                    <span className="text-[9px] text-muted-foreground whitespace-nowrap">{timeAgo(evt.timestamp)}</span>
                  </div>
                )
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
