import * as React from 'react'

import { cn } from '../../lib/cn'

export interface MessageBubbleProps extends React.HTMLAttributes<HTMLDivElement> {
  role: 'user' | 'assistant' | 'system'
  /**
   * `transcript` (default): Claude-like — assistant reads as plain prose; user is a subtle block.
   * `surface`: bordered cards with primary rail on assistant (dense / dashboards).
   */
  variant?: 'transcript' | 'surface'
  /** Provide a unique id for each message in a thread for ARIA relationships */
  messageId?: string
}

export function MessageBubble({
  className,
  role,
  children,
  variant = 'transcript',
  messageId,
  ...props
}: MessageBubbleProps) {
  const transcript =
    role === 'user'
      ? 'ml-auto max-w-[min(100%,36rem)] border border-border/40 bg-muted/50 px-3.5 py-2.5 text-sm leading-relaxed text-foreground shadow-none'
      : role === 'assistant'
        ? 'mr-auto max-w-[var(--chat-thread-max)] border-0 bg-transparent px-0 py-2 text-[0.9375rem] leading-[1.65] text-foreground shadow-none sm:py-2.5'
        : null

  const surface =
    role === 'user'
      ? 'ml-auto max-w-[var(--chat-thread-max)] border border-primary/22 bg-primary/[0.11] text-foreground shadow-[0_1px_2px_color-mix(in_srgb,var(--foreground)_6%,transparent)]'
      : role === 'assistant'
        ? 'mr-auto max-w-[var(--chat-thread-max)] border border-border/75 bg-card/80 text-card-foreground shadow-sm [border-left-width:3px] [border-left-color:color-mix(in_srgb,var(--primary)_52%,var(--border))]'
        : null

  const system = 'mx-auto border-dashed border-warning/50 bg-muted/40 text-muted-foreground'

  const tone =
    role === 'system'
      ? system
      : variant === 'surface'
        ? surface!
        : transcript!

  const padding =
    role === 'system'
      ? 'max-w-[var(--chat-thread-max)] px-3.5 py-2.5 sm:px-4 sm:py-3'
      : variant === 'surface'
        ? 'max-w-[var(--chat-thread-max)] px-3.5 py-2.5 text-sm leading-relaxed tracking-[0.01em] transition-colors duration-200 ease-smooth sm:px-4 sm:py-3 text-base sm:text-sm'
        : role === 'user'
          ? 'transition-colors duration-200 ease-smooth'
          : 'max-w-[var(--chat-thread-max)] transition-colors duration-200 ease-smooth'

  const roleLabel =
    role === 'user' ? 'User message' : role === 'assistant' ? 'Assistant response' : 'System message'

  return (
    <article
      className={cn(padding, tone, className)}
      role="article"
      aria-label={roleLabel}
      aria-roledescription={role === 'assistant' ? 'assistant message' : role === 'user' ? 'user message' : 'system message'}
      {...props}
    >
      {children}
    </article>
  )
}
