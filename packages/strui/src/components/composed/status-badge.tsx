'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

export interface StatusBadgeProps {
  /** Status label text */
  children: ReactNode
  /** Status tone */
  tone?: 'default' | 'success' | 'warning' | 'destructive' | 'info'
  /** Size variant */
  size?: 'sm' | 'md'
  /** Additional CSS classes */
  className?: string
}

const toneClasses = {
  default: 'bg-muted text-muted-foreground',
  success: 'bg-success/15 text-success',
  warning: 'bg-warning/15 text-warning',
  destructive: 'bg-destructive/15 text-destructive',
  info: 'bg-primary/15 text-primary',
}

const sizeClasses = {
  sm: 'text-[9px] px-1.5 py-0.5',
  md: 'text-[10px] px-2 py-0.5',
}

/**
 * Status badge with tone-based coloring.
 *
 * Renders a small pill-shaped badge with consistent styling
 * for status indicators, labels, and tags.
 *
 * @example
 * ```tsx
 * <StatusBadge tone="success">Active</StatusBadge>
 * <StatusBadge tone="warning" size="sm">Degraded</StatusBadge>
 * <StatusBadge tone="destructive">Error</StatusBadge>
 * ```
 */
export function StatusBadge({
  children,
  tone = 'default',
  size = 'sm',
  className,
}: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full font-medium',
        toneClasses[tone],
        sizeClasses[size],
        className
      )}
    >
      {children}
    </span>
  )
}
