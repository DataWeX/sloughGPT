'use client'

import { useCallback, useState } from 'react'

import { Button } from '../ui/button'
import { cn } from '../../lib/cn'

export interface CopyButtonProps {
  text: string
  labels?: [string, string]
  className?: string
  disabled?: boolean
}

/** Copies `text` to the clipboard; shows transient “Copied” feedback. */
export function CopyButton({
  text,
  labels = ['Copy', 'Copied'],
  className,
  disabled,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  const onClick = useCallback(async () => {
    if (disabled) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopied(false)
    }
  }, [disabled, text])

  return (
    <Button
      type="button"
      variant="secondary"
      size="sm"
      disabled={disabled}
      className={cn('font-mono text-xs', className)}
      onClick={onClick}
      aria-label={copied ? labels[1] : labels[0]}
      aria-pressed={copied}
    >
      <span className="sr-only">{copied ? 'Copied to clipboard' : 'Copy to clipboard'}</span>
      <span aria-hidden="true">{copied ? labels[1] : labels[0]}</span>
    </Button>
  )
}
