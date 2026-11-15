'use client'

import { cn } from '@sloughgpt/strui'

export interface MessageFrameProps {
  role: 'user' | 'assistant'
  children: React.ReactNode
  /** Metadata (role, model, metrics, timestamp) — lives INSIDE the frame. */
  footer?: React.ReactNode
  /** Editing needs the full column width (see MessageContent's edit form). */
  fullWidth?: boolean
  className?: string
}

/**
 * MessageFrame — the one surface every chat message renders through.
 *
 * Direction B (user ruling 2026-11-03, "visual consistency please"): user and
 * assistant share one shape (`rounded-2xl` — no pills) and only differ in
 * alignment and tint, and the metadata is a footer inside the frame, never a
 * detached right gutter. Renderers compose this; they never re-decide chrome.
 * Tokens only — primary tint for the user, card for the assistant, existing
 * spacing/type scale.
 */
export function MessageFrame({
  role,
  children,
  footer,
  fullWidth = false,
  className,
}: MessageFrameProps) {
  const isUser = role === 'user'
  return (
    <div
      data-message-frame={role}
      className={cn('flex min-w-0', isUser && 'justify-end', fullWidth && 'w-full', className)}
    >
      <div
        className={cn(
          'min-w-0 rounded-2xl border px-4 py-2.5',
          isUser
            ? 'bg-primary/10 border-primary/20 text-foreground max-w-[85%] sm:max-w-[75%] lg:max-w-[65%]'
            : 'bg-card border-border text-foreground max-w-[70ch]',
          fullWidth && 'w-full max-w-none',
        )}
      >
        {children}
        {footer && (
          <div
            data-message-frame-footer
            className={cn(
              'mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5',
              'select-none font-mono text-[10px] leading-5 text-muted-foreground/60',
            )}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
