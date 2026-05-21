import type { HTMLAttributes, ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ToolbarProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

/** Horizontal cluster of icon buttons or compact controls (filters, view toggles). */
export function Toolbar({ className, children, ...props }: ToolbarProps) {
  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-2 border-b border-border bg-card/30 px-3 py-2 sm:px-4',
        className,
      )}
      role="toolbar"
      {...props}
    >
      {children}
    </div>
  )
}
