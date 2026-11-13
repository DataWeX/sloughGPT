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
    <div className={cn('border border-border/40 rounded-lg bg-card overflow-hidden', className)} data-testid={testId}>
      <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-border/30 bg-muted/20">
        <span className="text-[10px] font-medium text-muted-foreground/60">{title}</span>
        {actions && <div className="flex items-center gap-0.5">{actions}</div>}
      </div>
      <div className="p-2.5">
        {children}
      </div>
    </div>
  )
}
