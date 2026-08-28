'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconCheck, IconDownload } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface CodeExecution {
  id: string
  code: string
  language: string
  output: string
  error?: string
  exitCode: number
  duration: number
  timestamp: Date
}

interface CodeExecutionResultsProps {
  executions: CodeExecution[]
  onRerun?: (id: string) => void
  className?: string
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function getStatusColor(exitCode: number): string {
  if (exitCode === 0) return 'text-success'
  return 'text-destructive'
}

function getStatusIcon(exitCode: number): React.ReactNode {
  if (exitCode === 0) return <IconCheck className="h-3.5 w-3.5" />
  return <IconX className="h-3.5 w-3.5" />
}

export const CodeExecutionResults = memo(function CodeExecutionResults({
  executions,
  onRerun,
  className,
}: CodeExecutionResultsProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const handleCopy = useCallback(async (code: string, id: string) => {
    await navigator.clipboard.writeText(code)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }, [])

  const handleToggle = useCallback((id: string) => {
    setExpandedId(prev => prev === id ? null : id)
  }, [])

  if (executions.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-4', className)}>
        No code executions
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      {executions.map(exec => (
        <div
          key={exec.id}
          className={cn(
            'border rounded-lg overflow-hidden',
            exec.exitCode === 0 ? 'border-success/30' : 'border-destructive/30',
          )}
        >
          <div
            className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-muted/30"
            onClick={() => handleToggle(exec.id)}
          >
            <div className="flex items-center gap-2">
              <span className={cn('text-xs font-mono', getStatusColor(exec.exitCode))}>
                {getStatusIcon(exec.exitCode)}
              </span>
              <span className="text-xs font-medium">{exec.language}</span>
              <span className="text-[10px] text-muted-foreground">
                {formatDuration(exec.duration)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">
                {new Date(exec.timestamp).toLocaleTimeString()}
              </span>
              <span className="text-xs">{expandedId === exec.id ? '▼' : '▶'}</span>
            </div>
          </div>

          {expandedId === exec.id && (
            <div className="border-t">
              <div className="relative">
                <pre className="p-3 text-xs font-mono bg-muted/50 overflow-x-auto max-h-[200px] overflow-y-auto">
                  {exec.code}
                </pre>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="absolute top-1 right-1 h-5 w-5"
                  onClick={() => handleCopy(exec.code, exec.id)}
                >
                  {copiedId === exec.id ? (
                    <IconCheck className="h-3 w-3" />
                  ) : (
                    <IconDownload className="h-3 w-3" />
                  )}
                </Button>
              </div>

              {exec.output && (
                <div className="border-t">
                  <div className="px-3 py-1 text-[10px] text-muted-foreground bg-muted/30">
                    Output
                  </div>
                  <pre className="p-3 text-xs font-mono bg-muted/20 overflow-x-auto max-h-[150px] overflow-y-auto">
                    {exec.output}
                  </pre>
                </div>
              )}

              {exec.error && (
                <div className="border-t">
                  <div className="px-3 py-1 text-[10px] text-destructive bg-destructive/10">
                    Error
                  </div>
                  <pre className="p-3 text-xs font-mono text-destructive bg-destructive/5 overflow-x-auto max-h-[150px] overflow-y-auto">
                    {exec.error}
                  </pre>
                </div>
              )}

              <div className="flex items-center justify-between px-3 py-2 border-t bg-muted/30">
                <span className="text-[10px] text-muted-foreground">
                  Exit code: {exec.exitCode}
                </span>
                {onRerun && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-[10px] h-5"
                    onClick={() => onRerun(exec.id)}
                  >
                    Rerun
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
})