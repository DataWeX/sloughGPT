import * as React from 'react'

import { Button } from '../ui/button'
import { Textarea } from '../ui/textarea'
import { cn } from '../../lib/cn'

/** Allows `data-*` test hooks without fighting DOM attribute typings. */
type DataAttributeProps = { [key: `data-${string}`]: string | undefined }

export interface PromptComposerProps extends Omit<React.FormHTMLAttributes<HTMLFormElement>, 'onSubmit'> {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  placeholder?: string
  sendLabel?: string
  disabled?: boolean
  /** Shown while a request is in flight; send button stays touch-sized. */
  busy?: boolean
  /** Forward to the textarea (focus management in host apps). */
  textareaRef?: React.Ref<HTMLTextAreaElement>
  /** Extra attributes for the textarea (e.g. `data-testid`). */
  textAreaProps?: React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
    'data-testid'?: string
  }
  /** When false, omits `str-safe-bottom` so the host can own safe-area padding. Default true. */
  safeAreaBottom?: boolean
  /** Extra props for the submit control (e.g. `data-testid` for E2E). */
  sendButtonProps?: React.ComponentPropsWithoutRef<typeof Button> & DataAttributeProps
}

/**
 * Bottom-stacked prompt area for mobile; row layout from `sm` and up.
 * Applies `str-safe-bottom` so content clears the iOS home indicator in standalone PWA.
 */
export function PromptComposer({
  className,
  value,
  onValueChange,
  onSubmit,
  placeholder = 'Message…',
  sendLabel = 'Send',
  disabled,
  busy,
  textareaRef,
  textAreaProps,
  safeAreaBottom = true,
  sendButtonProps,
  ...formProps
}: PromptComposerProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!disabled && !busy && value.trim()) onSubmit()
  }

  return (
    <form
      className={cn(
        safeAreaBottom && 'str-safe-bottom',
        'border-t border-border/80 bg-card/95 backdrop-blur-md',
        'p-3 sm:p-4',
        className,
      )}
      onSubmit={handleSubmit}
      role="form"
      aria-label="Message composer"
      aria-busy={busy}
      {...formProps}
    >
      <div className="flex flex-col gap-2.5 sm:flex-row sm:items-end">
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onValueChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || busy}
          rows={2}
          {...textAreaProps}
          className={cn('min-h-[5.25rem] flex-1 resize-y sm:min-h-[4.5rem]', textAreaProps?.className)}
          aria-label={textAreaProps?.['aria-label'] ?? placeholder}
          onKeyDown={(e) => {
            textAreaProps?.onKeyDown?.(e)
            if (e.defaultPrevented) return
            if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
              e.preventDefault()
              if (!disabled && !busy && value.trim()) onSubmit()
            }
          }}
        />
        <Button
          {...sendButtonProps}
          type="submit"
          disabled={disabled || busy || !value.trim()}
          className={cn('str-touch-target w-full shrink-0 sm:w-auto', sendButtonProps?.className)}
          aria-label={busy ? 'Sending message' : sendLabel}
          aria-busy={busy}
        >
          {busy ? (
            <span className="flex items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-current/30 border-t-current" aria-hidden="true" />
              <span>Sending</span>
            </span>
          ) : sendLabel}
        </Button>
      </div>
    </form>
  )
}
