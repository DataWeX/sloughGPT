import * as React from 'react'

import { cn } from '../../lib/cn'

export interface ChatThreadProps extends React.HTMLAttributes<HTMLDivElement> {
  /** When true, thread is announced politely for new messages (screen readers). */
  live?: boolean
  /** Vertical rhythm: comfortable (default) or compact for dense logs. */
  density?: 'comfortable' | 'compact'
}

/**
 * Scrollable message list. Uses `str-chat-thread` (flat notebook canvas) + `str-chat-scroll` + safe horizontal insets for notched phones / PWA.
 * Height comes from the parent (e.g. `flex-1 min-h-0` in a column shell); cap with `max-h-*` only when embedding in dashboards.
 * Pair with {@link PromptComposer} using `str-safe-bottom` so the composer clears the home indicator.
 */
export const ChatThread = React.forwardRef<HTMLDivElement, ChatThreadProps>(
  ({ className, children, live = true, density = 'comfortable', ...props }, ref) => (
    <section
      ref={ref}
      role="log"
      aria-live={live ? 'polite' : 'off'}
      aria-relevant="additions"
      aria-label="Chat messages"
      className={cn(
        'str-chat-thread str-chat-scroll str-safe-x flex min-h-0 flex-col overflow-y-auto',
        density === 'compact'
          ? 'gap-2.5 px-3 py-3 sm:gap-3 sm:px-4'
          : 'gap-5 px-3 py-4 sm:gap-6 sm:px-6 sm:py-6',
        className,
      )}
      {...props}
    >
      {children}
    </section>
  ),
)
ChatThread.displayName = 'ChatThread'
