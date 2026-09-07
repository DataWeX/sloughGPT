'use client'

import { cn } from '@sloughgpt/strui'

interface SectionLabelProps {
  children: React.ReactNode
  className?: string
}

export function SectionLabel({ children, className }: SectionLabelProps) {
  return (
    <span className={cn(
      'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block',
      className
    )}>
      {children}
    </span>
  )
}
