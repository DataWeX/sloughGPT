'use client'

import { cn, Button } from '@sloughgpt/strui'
import { IconSend, IconStop } from '@sloughgpt/strui'

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
        "h-11 w-11 shrink-0 p-0 flex items-center justify-center transition-all duration-200 rounded-xl",
        loading
          ? "bg-destructive hover:bg-destructive/90 text-destructive-foreground"
          : hasContent
            ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm"
            : "bg-muted/30 text-muted-foreground/40"
      )}
      aria-label={loading ? "Stop generation" : "Send message"}
      data-send-button="true"
    >
      {loading ? (
        <IconStop className="h-4 w-4" />
      ) : (
        <IconSend className="h-4 w-4" />
      )}
    </Button>
  )
}
