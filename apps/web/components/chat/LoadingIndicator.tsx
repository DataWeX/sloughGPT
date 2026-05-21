'use client'

import { TypingIndicator } from './TypingDots'

export function LoadingIndicator() {
  return (
    <div 
      className="flex justify-start"
      role="status"
      aria-label="Model is generating response"
      aria-live="polite"
    >
      <div className="rounded-2xl rounded-bl-sm bg-card px-4 py-3 text-sm text-muted-foreground border border-border/60 shadow-sm">
        <TypingIndicator label="Generating response" />
      </div>
    </div>
  )
}