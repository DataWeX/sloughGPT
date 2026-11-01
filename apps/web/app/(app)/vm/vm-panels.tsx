'use client'

/**
 * Small presentational panels shared by the VM console tabs: a label/value
 * row, the training-job card, and the VGA text-buffer screen.
 */
import { useState } from 'react'
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Progress } from '@sloughgpt/strui'
import { StatusBanner } from '@/components/composed/StatusBanner'
import type { VMTrainingJob } from '@/lib/vm-controller'

export function StatusRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  )
}

export function TrainingCard({ job, onStop }: { job: VMTrainingJob | null; onStop: () => void }) {
  if (!job) return null
  const running = ['running', 'queued', 'starting'].includes(job.status)
  const failed = ['failed', 'cancelled', 'not_found', 'error', 'interrupted'].includes(job.status)
  const pct: number | null = job.progress != null ? Math.round(job.progress * 100) : null
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Training</CardTitle>
          <div className="flex items-center gap-2">
            {running && (
              <Button size="sm" variant="ghost" onClick={onStop}>
                Stop
              </Button>
            )}
            <Badge
              variant={job.status === 'completed' ? 'success' : failed ? 'error' : 'warning'}
              size="sm"
            >
              {job.status}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <StatusRow label="Job" value={`#${job.job_id}`} />
        {job.api_job_id && <StatusRow label="API job" value={job.api_job_id} />}
        {running && (
          <>
            <Progress
              value={pct}
              variant={pct != null && pct > 0 ? 'default' : 'warning'}
              size="sm"
            />
            <p className="text-xs text-muted-foreground">
              Training in progress — {pct != null ? `${pct}%` : '--'}
            </p>
          </>
        )}
        {job.status === 'completed' && (
          <p className="text-xs text-success">Training completed successfully.</p>
        )}
        {job.status === 'completed' && job.result && (
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Result
            </p>
            <pre className="text-xs font-mono bg-muted/30 p-2 rounded overflow-x-auto whitespace-pre-wrap text-success">
              {job.result}
            </pre>
          </div>
        )}
        {job.error && <StatusBanner variant="error" message={job.error} dismissible={false} />}
      </CardContent>
    </Card>
  )
}

export function VGADisplay({
  text,
  cells,
}: {
  text?: string
  cells?: { ch: string; fg: string; bg: string }[]
}) {
  const [fullScreen, setFullScreen] = useState(false)

  // Render colored cells as rows of spans
  const renderCells = () => {
    if (!cells) return null
    const rows: React.ReactNode[] = []
    for (let row = 0; row < 25; row++) {
      const rowCells = cells.slice(row * 80, (row + 1) * 80)
      // Skip empty rows
      const hasContent = rowCells.some((c) => c.ch !== ' ')
      if (!hasContent) continue
      rows.push(
        <div key={row}>
          {rowCells.map((c, col) => (
            <span key={col} style={{ color: c.fg, backgroundColor: c.bg }}>
              {c.ch}
            </span>
          ))}
        </div>,
      )
    }
    return rows
  }

  return (
    <Card className={fullScreen ? 'fixed inset-4 z-50' : ''}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Screen (VGA 0xB8000)</CardTitle>
          <Button size="sm" variant="ghost" onClick={() => setFullScreen(!fullScreen)}>
            {fullScreen ? 'Exit' : 'Fullscreen'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div
          className={`bg-black font-mono text-xs p-3 rounded overflow-y-auto whitespace-pre ${
            fullScreen ? 'h-[calc(100dvh-8rem)]' : 'h-40'
          }`}
        >
          {cells ? (
            renderCells()
          ) : text ? (
            <span className="text-success">{text}</span>
          ) : (
            <span className="text-success">Programs that write to 0xB8000 will appear here.</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
