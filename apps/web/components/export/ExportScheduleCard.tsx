'use client'

import { useState, useEffect, useCallback } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'

interface ExportSchedule {
  id: string
  name: string
  type: 'model' | 'training-data' | 'checkpoints'
  format: string
  frequency: 'daily' | 'weekly' | 'monthly'
  enabled: boolean
  lastRun?: number
  nextRun: number
  timestamp: number
}

const STORAGE_KEY = 'sloughgpt-export-schedules'

async function loadSchedules(): Promise<ExportSchedule[]> {
  try {
    const entry = await chatDB.getKV<ExportSchedule[]>(STORAGE_KEY)
    return entry ?? []
  } catch { return [] }
}

async function saveSchedules(schedules: ExportSchedule[]) {
  try { await chatDB.setKV(STORAGE_KEY, schedules) } catch { /* quota exceeded */ }
}

function calcNextRun(frequency: string, from: number): number {
  const d = new Date(from)
  if (frequency === 'daily') d.setDate(d.getDate() + 1)
  else if (frequency === 'weekly') d.setDate(d.getDate() + 7)
  else if (frequency === 'monthly') d.setMonth(d.getMonth() + 1)
  return d.getTime()
}

function formatNextRun(ts: number): string {
  const now = Date.now()
  const diff = ts - now
  if (diff < 0) return 'Overdue'
  if (diff < 60 * 60 * 1000) return `In ${Math.ceil(diff / 60000)}m`
  if (diff < 24 * 60 * 60 * 1000) return `In ${Math.ceil(diff / 3600000)}h`
  return `In ${Math.ceil(diff / 86400000)}d`
}

const FREQUENCY_OPTIONS = ['daily', 'weekly', 'monthly'] as const
const TYPE_OPTIONS = [
  { value: 'model', label: 'Model' },
  { value: 'training-data', label: 'Training Data' },
  { value: 'checkpoints', label: 'Checkpoints' },
] as const

export function ExportScheduleCard() {
  const [schedules, setSchedules] = useState<ExportSchedule[]>([])
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editType, setEditType] = useState<ExportSchedule['type']>('model')
  const [editFormat, setEditFormat] = useState('sou')
  const [editFrequency, setEditFrequency] = useState<ExportSchedule['frequency']>('weekly')

  useEffect(() => { loadSchedules().then(setSchedules) }, [])

  const handleSave = useCallback(() => {
    if (!editName.trim()) return
    const now = Date.now()
    const schedule: ExportSchedule = {
      id: `sched-${Date.now()}`,
      name: editName.trim(),
      type: editType,
      format: editFormat,
      frequency: editFrequency,
      enabled: true,
      nextRun: calcNextRun(editFrequency, now),
      timestamp: now,
    }
    const updated = [...schedules, schedule]
    setSchedules(updated)
    saveSchedules(updated).catch(() => {})
    setEditing(false)
    setEditName('')
  }, [editName, editType, editFormat, editFrequency, schedules])

  const toggleEnabled = useCallback((id: string) => {
    const updated = schedules.map(s =>
      s.id === id ? { ...s, enabled: !s.enabled } : s
    )
    setSchedules(updated)
    saveSchedules(updated).catch(() => {})
  }, [schedules])

  const deleteSchedule = useCallback((id: string) => {
    const updated = schedules.filter(s => s.id !== id)
    setSchedules(updated)
    saveSchedules(updated).catch(() => {})
  }, [schedules])

  return (
    <Card data-testid="export-schedule">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Scheduled Exports</CardTitle>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setEditing(!editing)}>
            {editing ? 'Cancel' : '+ New'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {editing && (
          <div className="space-y-2 rounded-md border border-border/40 p-2.5">
            <Input
              value={editName}
              onChange={e => setEditName(e.target.value)}
              placeholder="Schedule name"
              className="h-7 text-[11px]"
            />
            <div className="grid grid-cols-3 gap-2">
              <select
                className="text-[10px] border border-border/40 rounded px-1.5 py-1 bg-background"
                value={editType}
                onChange={e => setEditType(e.target.value as ExportSchedule['type'])}
              >
                {TYPE_OPTIONS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
              <select
                className="text-[10px] border border-border/40 rounded px-1.5 py-1 bg-background"
                value={editFormat}
                onChange={e => setEditFormat(e.target.value)}
              >
                {['sou', 'onnx', 'jsonl', 'gguf'].map(f => <option key={f} value={f}>{f.toUpperCase()}</option>)}
              </select>
              <select
                className="text-[10px] border border-border/40 rounded px-1.5 py-1 bg-background"
                value={editFrequency}
                onChange={e => setEditFrequency(e.target.value as ExportSchedule['frequency'])}
              >
                {FREQUENCY_OPTIONS.map(f => <option key={f} value={f}>{f.charAt(0).toUpperCase() + f.slice(1)}</option>)}
              </select>
            </div>
            <Button size="sm" className="h-6 text-[10px]" onClick={handleSave} disabled={!editName.trim()}>
              Save Schedule
            </Button>
          </div>
        )}

        {schedules.length === 0 && !editing ? (
          <p className="text-xs text-muted-foreground text-center py-2">No scheduled exports.</p>
        ) : (
          <div className="space-y-1.5">
            {schedules.map(s => (
              <div
                key={s.id}
                className={cn(
                  'flex items-center justify-between rounded-md border px-2.5 py-2 text-[11px] transition-colors group',
                  s.enabled ? 'border-border/40 hover:bg-muted/20' : 'border-border/20 opacity-60',
                )}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate">{s.name}</span>
                    <span className="text-[9px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
                      {s.type}
                    </span>
                    <span className="text-[9px] text-muted-foreground">{s.frequency}</span>
                  </div>
                  <div className="text-[9px] text-muted-foreground mt-0.5">
                    {s.format.toUpperCase()} · {s.enabled ? formatNextRun(s.nextRun) : 'Paused'}
                    {s.lastRun && <> · Last: {new Date(s.lastRun).toLocaleDateString()}</>}
                  </div>
                </div>
                <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button size="sm" variant="ghost" className="h-5 text-[9px]" onClick={() => toggleEnabled(s.id)}>
                    {s.enabled ? 'Pause' : 'Resume'}
                  </Button>
                  <Button size="sm" variant="ghost" className="h-5 text-[9px] text-destructive" onClick={() => deleteSchedule(s.id)}>
                    Del
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
