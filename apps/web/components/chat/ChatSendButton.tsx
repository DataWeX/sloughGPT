'use client'

import { Button } from '@/components/ui/button'
import { IconSend } from '@/components/ui'
import { cn } from '@/lib/cn'

interface ChatSendButtonProps {
  loading: boolean
  hasContent: boolean
  onSend: () => void
  onStop?: () => void
  disabled: boolean
}

export function ChatSendButton({ loading, hasContent, onSend, onStop, disabled }: ChatSendButtonProps) {
  const isDisabled = !loading && (disabled || !hasContent)
  return (
    <Button
      onClick={loading ? onStop : onSend}
      disabled={isDisabled}
      className={cn(
        "h-10 w-12 shrink-0 p-0 flex items-center justify-center transition-all",
        loading
          ? "bg-destructive hover:bg-destructive/90 text-destructive-foreground"
          : "bg-primary text-primary-foreground hover:opacity-90"
      )}
      aria-label={loading ? "Stop generation" : "Send message"}
      data-send-button="true"
    >
      {loading ? (
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 6h12v12H6z" />
        </svg>
      ) : (
        <IconSend className="h-4 w-4" />
      )}
    </Button>
  )
}
