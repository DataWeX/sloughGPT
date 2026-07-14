'use client'

import { useServerOutput } from '@/hooks/useServerOutput'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconDownload } from '@sloughgpt/strui'

interface OutputCardProps {
  title?: string
  height?: string
  tail?: number
  maxLines?: number
}

export function OutputCard({ title = 'Server Output', height = 'h-[180px]', tail, maxLines }: OutputCardProps) {
  const { lines, streaming, clear, scrollRef, paused, togglePause, exportLines } = useServerOutput({ tail, maxLines })

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{title}</CardTitle>
          <div className="flex items-center gap-2">
            <span className={`inline-block w-2 h-2 rounded-full ${paused ? 'bg-warning' : streaming ? 'bg-success animate-pulse' : 'bg-muted-foreground/50'}`} />
            <span className="text-xs text-muted-foreground">{paused ? 'Paused' : streaming ? 'Live' : 'Off'}</span>
            <Button variant="ghost" size="sm" onClick={togglePause} aria-label={paused ? 'Resume output' : 'Pause output'}>
              {paused ? '▶' : '⏸'}
            </Button>
            {lines.length > 0 && (
              <>
                <Button variant="ghost" size="sm" onClick={() => exportLines('text')} aria-label="Export as log file">
                  <IconDownload className="h-3 w-3" />
                </Button>
                <Button variant="ghost" size="sm" onClick={clear} aria-label="Clear output">
                  Clear
                </Button>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={scrollRef}
          className={`${height} overflow-y-auto font-mono text-xs bg-background border rounded-lg p-2 space-y-0.5`}
          role="log"
          aria-label="Server output"
        >
          {lines.length === 0 ? (
            <div className="text-muted-foreground py-4 text-center text-xs">
              {streaming ? 'Waiting for output...' : 'Output will appear here during server activity'}
            </div>
          ) : (
            lines.map((line, i) => (
              <div key={`${line.ts}-${i}`} className="flex gap-2 leading-tight">
                <span className="text-muted-foreground shrink-0 w-14">
                  {new Date(line.ts * 1000).toLocaleTimeString()}
                </span>
                <span className={`shrink-0 w-10 ${
                  line.level === 'error' ? 'text-destructive' :
                  line.level === 'warning' ? 'text-warning' :
                  'text-muted-foreground'
                }`}>
                  {line.level}
                </span>
                <span className="flex-1 min-w-0 break-all">{line.text}</span>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
