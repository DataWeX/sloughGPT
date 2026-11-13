'use client'

import { cn } from '../../lib/cn'

interface DividerProps {
  label?: string
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

export function Divider({ label, orientation = 'horizontal', className }: DividerProps) {
  if (orientation === 'vertical') {
    return <div className={cn('w-px self-stretch bg-border', className)} />
  }

  if (label) {
    return (
      <div className={cn('flex items-center gap-3', className)} role="separator">
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground whitespace-nowrap">{label}</span>
        <div className="flex-1 h-px bg-border" />
      </div>
    )
  }
  return <div className={cn('h-px bg-border', className)} role="separator" />
}
