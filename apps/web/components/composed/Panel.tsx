'use client'

import { cn } from '@sloughgpt/strui'

interface PanelProps {
  title: string
  actions?: React.ReactNode
  children: React.ReactNode
  className?: string
  testId?: string
}

export function Panel({ title, actions, children, className, testId }: PanelProps) {
  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)} data-testid={testId}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">{title}</span>
        {actions && <div className="flex items-center gap-1">{actions}</div>}
      </div>
      <div className="p-3">
        {children}
      </div>
    </div>
  )
}
