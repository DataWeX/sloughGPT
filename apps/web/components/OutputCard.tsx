'use client'

import React from 'react'
import { useServerOutput } from '@/hooks/useServerOutput'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconDownload } from '@sloughgpt/strui'
import type { OutputLine } from '@/lib/system-controller'

interface OutputCardProps {
  title?: string
  height?: string
  tail?: number
  maxLines?: number
  compact?: boolean
}

const LEVEL_COLOR: Record<string, string> = {
  error: 'rgb(var(--destructive))',
  critical: 'rgb(var(--destructive))',
  warning: 'rgb(var(--warning))',
  debug: 'rgb(var(--muted-foreground))',
}

const LEVEL_ABBR: Record<string, string> = {
  info: 'INF',
  error: 'ERR',
  warning: 'WRN',
  debug: 'DBG',
  critical: 'CRI',
}

const TAG_COLOR: Record<string, string> = {
  START: 'rgb(var(--primary))',
  REQ: 'rgb(var(--muted-foreground))',
  INFRA: 'rgb(var(--muted-foreground))',
  MODEL: 'rgb(var(--success))',
  SOUL: 'rgb(var(--success))',
  INF: 'rgb(var(--muted-foreground))',
  INFO: 'rgb(var(--muted-foreground))',
  WEB: 'rgb(var(--muted-foreground))',
}

function formatTs(ts: number): string {
  const d = new Date(ts * 1000)
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  const s = String(d.getSeconds()).padStart(2, '0')
  return `${h}:${m}:${s}`
}

const LogLine = React.memo(function LogLine({ line }: { line: OutputLine }) {
  const ts = formatTs(line.ts)
  const lvl = LEVEL_ABBR[line.level] ?? 'INF'
  const lvlColor = LEVEL_COLOR[line.level] ?? 'rgb(var(--muted-foreground))'
  const tagColor = line.tag ? (TAG_COLOR[line.tag] ?? 'rgb(var(--muted-foreground))') : undefined

  return (
    <div className="flex font-mono text-[11px] leading-5 py-[1px]">
      <span className="shrink-0 w-[58px] sm:w-[70px] text-muted-foreground/50 tabular-nums">{ts}</span>
      <span className="shrink-0 w-[32px] text-center font-semibold" style={{ color: lvlColor }}>{lvl}</span>
      {line.tag ? (
        <span className="shrink-0 w-[56px] text-center font-medium" style={{ color: tagColor }}>[{line.tag}]</span>
      ) : (
        <span className="shrink-0 w-[56px]" />
      )}
      {line.source ? (
        <span className="shrink-0 text-muted-foreground/30 truncate max-w-[96px] sm:max-w-[160px] pr-2">{line.source}</span>
      ) : (
        <span className="shrink-0 w-[96px] sm:w-[160px]" />
      )}
      <span className={`flex-1 min-w-0 break-all ${line.level === 'error' || line.level === 'critical' ? 'text-destructive' : line.level === 'warning' ? 'text-warning' : ''}`}>
        {line.text}
      </span>
    </div>
  )
})

export function OutputCard({ title = 'Server Output', height, tail, maxLines, compact }: OutputCardProps) {
  const { lines, streaming, clear, scrollRef, paused, togglePause, exportLines } = useServerOutput({ tail, maxLines })
  const h = height ?? (compact ? 'h-[220px]' : 'h-[280px]')

  const controls = (
    <div className="flex items-center gap-2">
      <span className={`inline-block w-2 h-2 rounded-full ${paused ? 'bg-warning' : streaming ? 'bg-success animate-pulse' : 'bg-muted-foreground/50'}`} />
      <span className="text-[11px] text-muted-foreground font-mono">{paused ? 'Paused' : streaming ? 'Live' : 'Off'}</span>
      <Button variant="ghost" size="sm" className="h-7 text-[10px]" onClick={togglePause} aria-label={paused ? 'Resume output' : 'Pause output'}>
        {paused ? '▶' : '⏸'}
      </Button>
      {lines.length > 0 && (
        <>
          <Button variant="ghost" size="sm" className="h-5 text-[10px]" onClick={() => exportLines('text')} aria-label="Export as log file">
            <IconDownload className="h-2.5 w-2.5" />
          </Button>
          <Button variant="ghost" size="sm" className="h-5 text-[10px]" onClick={clear} aria-label="Clear output">
            Clear
          </Button>
        </>
      )}
    </div>
  )

  return (
    <Card className={compact ? 'p-3' : ''}>
      {compact ? (
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</span>
          {controls}
        </div>
      ) : (
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">{title}</CardTitle>
            {controls}
          </div>
        </CardHeader>
      )}
      <CardContent>
        <div
          ref={scrollRef}
          className={`${h} overflow-y-auto rounded-lg border bg-zinc-950 text-zinc-300 p-3`}
          role="log"
          aria-label="Server output"
        >
          {lines.length === 0 ? (
            <div className="text-zinc-500 py-4 text-center text-xs">
              {streaming ? 'Waiting for output...' : 'Output will appear here during server activity'}
            </div>
          ) : (
            lines.map((line, i) => <LogLine key={`${line.ts}-${i}`} line={line} />)
          )}
        </div>
      </CardContent>
    </Card>
  )
}
