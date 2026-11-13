import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { join } from 'path'

function findRepoRoot(): string {
  let dir = process.cwd()
  while (dir !== '/') {
    if (existsSync(join(dir, '.kanban', 'board.jsonl')) && !dir.includes('.next')) return dir
    dir = join(dir, '..')
  }
  return process.cwd()
}

const REPO_ROOT = findRepoRoot()

function calendarDir(workspaceId?: string): string {
  if (workspaceId) return join(REPO_ROOT, '.calendar', workspaceId)
  return join(REPO_ROOT, '.calendar')
}

function eventsPath(workspaceId?: string): string {
  return join(calendarDir(workspaceId), 'events.jsonl')
}

export interface CalendarEvent {
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

export function readEvents(workspaceId?: string): CalendarEvent[] {
  const ep = eventsPath(workspaceId)
  if (!existsSync(ep)) return []
  const raw = readFileSync(ep, 'utf-8')
  const lines = raw.split('\n').filter(Boolean)
  const events: CalendarEvent[] = []
  for (const line of lines) {
    try {
      const obj = JSON.parse(line)
      if (obj.id && obj.title && obj.date) {
        events.push(obj as CalendarEvent)
      }
    } catch {
      // skip malformed lines
    }
  }
  return events
}

export function writeEvents(events: CalendarEvent[], workspaceId?: string): void {
  const ep = eventsPath(workspaceId)
  const dir = calendarDir(workspaceId)
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  const lines = events.map((event) => JSON.stringify(event))
  writeFileSync(ep, lines.join('\n') + '\n', 'utf-8')
}

export function createEvent(
  data: {
    title: string
    description?: string
    date: string
    start_time?: string
    end_time?: string
    color?: string
  },
  workspaceId?: string,
): CalendarEvent {
  const events = readEvents(workspaceId)
  const now = new Date().toISOString()
  const event: CalendarEvent = {
    id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: data.title,
    description: data.description || '',
    date: data.date,
    start_time: data.start_time || '09:00',
    end_time: data.end_time || '10:00',
    color: data.color || 'primary',
    created_at: now,
    updated_at: now,
  }
  events.push(event)
  writeEvents(events, workspaceId)
  return event
}

export function getEventsByDate(date: string, workspaceId?: string): CalendarEvent[] {
  const events = readEvents(workspaceId)
  return events.filter((e) => e.date === date)
}

export function deleteEvent(id: string, workspaceId?: string): boolean {
  const events = readEvents(workspaceId)
  const idx = events.findIndex((e) => e.id === id)
  if (idx === -1) return false
  events.splice(idx, 1)
  writeEvents(events, workspaceId)
  return true
}
