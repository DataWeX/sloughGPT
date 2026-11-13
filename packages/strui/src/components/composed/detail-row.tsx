'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface DetailRowProps {
  /** Label text */
  label: string
  /** Value content */
  value: ReactNode
  /** Render as monospace (for IDs, paths, etc.) */
  mono?: boolean
  /** Render as a link */
  href?: string
  /** Additional CSS classes for the row */
  className?: string
  /** Additional CSS classes for the label */
  labelClassName?: string
  /** Additional CSS classes for the value */
  valueClassName?: string
}

/**
 * Single key-value detail row.
 *
 * Renders a label and value in a horizontal layout with consistent
 * spacing and typography. Used in detail panels, settings, and
 * metadata displays.
 *
 * @example
 * ```tsx
 * <DetailRow label="Model" value="GPT-2" />
 * <DetailRow label="Path" value="/models/gpt2" mono />
 * <DetailRow label="Size" value="500 MB" valueClassName="text-muted-foreground" />
 * ```
 */
export function DetailRow({
  label,
  value,
  mono = false,
  href,
  className,
  labelClassName,
  valueClassName,
}: DetailRowProps) {
  const valueContent = href ? (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary hover:underline"
    >
      {value}
    </a>
  ) : (
    value
  )

  return (
    <div className={cn('flex items-center justify-between gap-2 text-xs', className)}>
      <span className={cn('text-muted-foreground shrink-0', labelClassName)}>
        {label}
      </span>
      <span className={cn('min-w-0 truncate text-right', mono && 'font-mono text-[10px]', valueClassName)}>
        {valueContent}
      </span>
    </div>
  )
}
