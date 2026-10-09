'use client'

import { cn } from '@sloughgpt/strui'

interface SectionLabelProps {
  children: React.ReactNode
  className?: string
}

export function SectionLabel({ children, className }: SectionLabelProps) {
  return (
    <span className={cn(
      'text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider mb-1.5 block',
      className
    )}>
      {children}
    </span>
  )
}
