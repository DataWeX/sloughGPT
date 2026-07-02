import type { HTMLAttributes, ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ScrollPanelProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

/**
 * Bounded scroll region with lattice-friendly border — logs, JSON, long option lists.
 * Uses **str-chat-scroll** for touch / PWA.
 */
export function ScrollPanel({ className, children, ...props }: ScrollPanelProps) {
  return (
    <div
      className={cn(
        'str-chat-scroll max-h-[min(50dvh,28rem)] overflow-y-auto rounded-none border border-border bg-card/30',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  )
}
