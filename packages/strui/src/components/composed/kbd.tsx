import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface KbdProps {
  children: ReactNode
  className?: string
}

/** Styled keyboard hint for shortcuts and command palettes. */
export function Kbd({ children, className }: KbdProps) {
  return (
    <kbd
      className={cn(
        'inline-flex items-center rounded-none border border-border bg-muted/80 px-1.5 py-0.5 font-mono text-[0.65rem] font-medium text-muted-foreground',
        className,
      )}
    >
      {children}
    </kbd>
  )
}
