import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ChatLayoutProps {
  /** Sticky header (title, model picker, token meter). */
  header?: ReactNode
  /** Scrollable message region — usually **ChatThread**. */
  thread: ReactNode
  /** Fixed bottom — usually **PromptComposer**. */
  composer: ReactNode
  className?: string
}

/**
 * Opinionated full-height chat column: header + flex-1 thread + composer.
 * Uses `min-h-0` so nested scroll regions work inside flex layouts.
 */
export function ChatLayout({ header, thread, composer, className }: ChatLayoutProps) {
  return (
    <div className={cn('flex str-min-h-screen flex-col bg-background', className)}>
      {header ? <div className="shrink-0">{header}</div> : null}
      <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">{thread}</div>
      <div className="relative z-[1] shrink-0 shadow-[0_-10px_32px_-16px_color-mix(in_srgb,var(--foreground)_12%,transparent)]">
        {composer}
      </div>
    </div>
  )
}
