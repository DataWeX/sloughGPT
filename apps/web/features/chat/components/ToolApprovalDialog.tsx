'use client'

import { useState, useCallback, memo } from 'react'
import { cn } from '@sloughgpt/strui'

export interface ToolApprovalDialogProps {
  toolName: string
  args?: Record<string, unknown>
  onApprove: (approved: boolean) => void
  className?: string
}

export const ToolApprovalDialog = memo(function ToolApprovalDialog({
  toolName,
  args,
  onApprove,
  className,
}: ToolApprovalDialogProps) {
  const [decided, setDecided] = useState(false)

  const handleApprove = useCallback(() => {
    setDecided(true)
    onApprove(true)
  }, [onApprove])

  const handleDeny = useCallback(() => {
    setDecided(true)
    onApprove(false)
  }, [onApprove])

  if (decided) {
    return null
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3 p-3 rounded-lg border border-border bg-card',
        className
      )}
      role="alert"
      aria-label={`Tool approval required: ${toolName}`}
    >
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-foreground truncate">
          {toolName}
        </p>
        {args && Object.keys(args).length > 0 && (
          <p className="text-xs text-muted-foreground mt-0.5 truncate">
            {Object.entries(args).map(([k, v]) => `${k}: ${String(v)}`).join(', ')}
          </p>
        )}
      </div>
      <div className="flex gap-2 shrink-0">
        <button
          type="button"
          onClick={handleDeny}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-muted text-muted-foreground hover:bg-muted/80 transition-colors"
        >
          Deny
        </button>
        <button
          type="button"
          onClick={handleApprove}
          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Approve
        </button>
      </div>
    </div>
  )
})
