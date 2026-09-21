'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, Button, Badge } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface CalendarEvent {
  id: string
  title: string
  description: string
  date: string
  start_time: string
  end_time: string
  color: string
  created_at: string
  updated_at: string
}

const HOURS = Array.from({ length: 24 }, (_, i) => i)

const EVENT_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  primary: {
    bg: 'bg-primary/10',
    border: 'border-l-primary',
    text: 'text-primary',
  },
  accent: {
    bg: 'bg-accent/10',
    border: 'border-l-accent',
    text: 'text-accent',
  },
  success: {
    bg: 'bg-success/10',
    border: 'border-l-success',
    text: 'text-success',
  },
  warning: {
    bg: 'bg-warning/10',
    border: 'border-l-warning',
    text: 'text-warning',
  },
  destructive: {
    bg: 'bg-destructive/10',
    border: 'border-l-destructive',
    text: 'text-destructive',
  },
}

function formatHour(hour: number): string {
  if (hour === 0) return '12 AM'
  if (hour === 12) return '12 PM'
  if (hour < 12) return `${hour} AM`
  return `${hour - 12} PM`
}

function formatDate(date: Date): string {
  return date.toISOString().split('T')[0]
}

function getRelativeDay(date: Date): string {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const target = new Date(date)
  target.setHours(0, 0, 0, 0)
  const diffDays = Math.round((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24))

  if (diffDays === 0) return 'Today'
  if (diffDays === 1) return 'Tomorrow'
  if (diffDays === -1) return 'Yesterday'
  return date.toLocaleDateString('en-US', { weekday: 'long' })
}

export default function CalendarPage() {
  // Null until mount — a new Date() initial would differ between server
  // render and hydration, producing React #418 mismatches on every load.
  const [currentDate, setCurrentDate] = useState<Date | null>(null)
  const [todayStr, setTodayStr] = useState<string | null>(null)
  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const now = new Date()
    setCurrentDate(now)
    setTodayStr(formatDate(now))
  }, [])

  const fetchEvents = useCallback(async () => {
    if (!currentDate) {
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      const dateStr = formatDate(currentDate)
      const res = await fetch(`/api/calendar/events?date=${dateStr}`)
      const data = await res.json()
      setEvents(data.events || [])
    } catch (err) {
      console.error('Failed to fetch events:', err)
    } finally {
      setLoading(false)
    }
  }, [currentDate])

  useEffect(() => {
    fetchEvents()
  }, [fetchEvents])

  const navigateDay = (offset: number) => {
    if (!currentDate) return
    const newDate = new Date(currentDate)
    newDate.setDate(newDate.getDate() + offset)
    setCurrentDate(newDate)
  }

  const goToToday = () => setCurrentDate(new Date())

  const getEventsForHour = (hour: number) => {
    return events.filter((event) => {
      const eventHour = parseInt(event.start_time.split(':')[0], 10)
      return eventHour === hour
    })
  }

  const dayLabel = currentDate ? getRelativeDay(currentDate) : '—'
  const dateLabel = currentDate
    ? currentDate.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      })
    : '—'

  return (
    <PageContainer
      title={
        <div className="flex items-center gap-3">
          <span className="sl-h1">Calendar</span>
          <Badge variant="primary" size="sm">
            {dayLabel}
          </Badge>
        </div>
      }
      subtitle={dateLabel}
      headerRight={
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={goToToday} className="text-xs">
            Today
          </Button>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigateDay(-1)}
              className="h-8 w-8 p-0"
              aria-label="Previous day"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigateDay(1)}
              className="h-8 w-8 p-0"
              aria-label="Next day"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </Button>
          </div>
        </div>
      }
      loading={loading}
      loadingCards={4}
      maxWidth="max-w-5xl"
    >
      <div className="space-y-2">
        {HOURS.map((hour) => {
          const hourEvents = getEventsForHour(hour)
          const isCurrentHour =
            currentDate !== null &&
            todayStr !== null &&
            currentDate.getHours() === hour &&
            formatDate(currentDate) === todayStr

          return (
            <div
              key={hour}
              className={cn(
                'group flex min-h-[48px] border-b transition-colors',
                isCurrentHour
                  ? 'border-primary/30 bg-primary/5'
                  : 'border-border/50 hover:bg-muted/30',
              )}
            >
              <div className="w-16 shrink-0 py-2 pr-3 text-right">
                <span
                  className={cn(
                    'text-xs font-medium',
                    isCurrentHour ? 'text-primary' : 'text-muted-foreground',
                  )}
                >
                  {formatHour(hour)}
                </span>
              </div>
              <div className="flex-1 py-1 pl-3">
                {hourEvents.length > 0 ? (
                  <div className="space-y-1">
                    {hourEvents.map((event) => {
                      const colors = EVENT_COLORS[event.color] || EVENT_COLORS.primary
                      return (
                        <Card
                          key={event.id}
                          className={cn('border-l-[3px] py-2', colors.border, colors.bg)}
                        >
                          <CardContent className="p-2">
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0 flex-1">
                                <h4 className={cn('text-sm font-medium truncate', colors.text)}>
                                  {event.title}
                                </h4>
                                {event.description && (
                                  <p className="mt-0.5 text-xs text-muted-foreground truncate">
                                    {event.description}
                                  </p>
                                )}
                              </div>
                              <span className="shrink-0 text-[10px] text-muted-foreground">
                                {event.start_time} - {event.end_time}
                              </span>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                ) : (
                  <div className="h-full min-h-[32px] rounded-md border border-dashed border-transparent group-hover:border-border/50 transition-colors" />
                )}
              </div>
            </div>
          )
        })}
      </div>

      {events.length === 0 && !loading && (
        <Card className="mt-4">
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <div className="text-4xl mb-3 opacity-50">📅</div>
            <h3 className="text-sm font-medium text-foreground">No events scheduled</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              This day is clear. Use the API to add events.
            </p>
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
