'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

/* ── Avatar ─────────────────────────────────────────────────── */

interface AvatarProps {
  src?: string
  alt?: string
  fallback: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

export function Avatar({ src, alt, fallback, size = 'md', className }: AvatarProps) {
  const sizes = {
    xs: 'h-5 w-5 text-[9px]',
    sm: 'h-6 w-6 text-[10px]',
    md: 'h-8 w-8 text-xs',
    lg: 'h-10 w-10 text-sm',
    xl: 'h-12 w-12 text-base',
  }

  return (
    <div
      className={cn(
        'relative rounded-full overflow-hidden bg-primary/10 flex items-center justify-center shrink-0 ring-1 ring-border/40',
        sizes[size],
        className,
      )}
    >
      {src ? (
        <img src={src} alt={alt ?? fallback} className="h-full w-full object-cover" />
      ) : (
        <span className="font-semibold text-primary select-none">{fallback.slice(0, 2).toUpperCase()}</span>
      )}
    </div>
  )
}

/* ── Avatar Group ───────────────────────────────────────────── */

interface AvatarGroupProps {
  avatars: { src?: string; alt?: string; fallback: string }[]
  max?: number
  size?: 'xs' | 'sm' | 'md' | 'lg'
  className?: string
}

export function AvatarGroup({ avatars, max = 4, size = 'sm', className }: AvatarGroupProps) {
  const display = avatars.slice(0, max)
  const remaining = avatars.length - max

  const offsetMap = { xs: '-ml-1.5', sm: '-ml-2', md: '-ml-3', lg: '-ml-4' }
  const offset = offsetMap[size]

  return (
    <div className={cn('flex items-center', className)}>
      {display.map((av, i) => (
        <Avatar
          key={i}
          src={av.src}
          alt={av.alt}
          fallback={av.fallback}
          size={size}
          className={cn(i > 0 && offset, 'border-2 border-background')}
        />
      ))}
      {remaining > 0 && (
        <div
          className={cn(
            'rounded-full bg-muted flex items-center justify-center text-[10px] font-medium text-muted-foreground border-2 border-background',
            offset,
            size === 'xs' ? 'h-5 w-5' : size === 'sm' ? 'h-6 w-6' : size === 'md' ? 'h-8 w-8' : 'h-10 w-10',
          )}
        >
          +{remaining}
        </div>
      )}
    </div>
  )
}

/* ── Progress Bar ───────────────────────────────────────────── */

interface ProgressBarProps {
  value: number
  max?: number
  label?: string
  showValue?: boolean
  /** Size of the track */
  size?: 'xs' | 'sm' | 'default' | 'lg'
  variant?: 'default' | 'success' | 'warning' | 'error'
  animated?: boolean
  className?: string
}

export function ProgressBar({
  value,
  max = 100,
  label,
  showValue = true,
  size = 'default',
  variant = 'default',
  animated = false,
  className,
}: ProgressBarProps) {
  const percent = Math.min(Math.max((value / max) * 100, 0), 100)

  // Use design tokens, not raw Tailwind colors
  const trackColors = {
    default: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    error: 'bg-destructive',
  }

  const trackHeights = {
    xs: 'h-1',
    sm: 'h-1.5',
    default: 'h-2',
    lg: 'h-3',
  }

  return (
    <div className={cn('space-y-1', className)}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs text-muted-foreground">
          {label && <span>{label}</span>}
          {showValue && <span className="font-medium text-foreground">{Math.round(percent)}%</span>}
        </div>
      )}
      <div
        className={cn('w-full rounded-full overflow-hidden bg-muted', trackHeights[size])}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label}
      >
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500 ease-smooth',
            trackColors[variant],
            animated && 'animate-pulse',
          )}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}

/* ── Spinner ────────────────────────────────────────────────── */

interface SpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg'
  className?: string
}

export function Spinner({ size = 'md', className }: SpinnerProps) {
  const sizes = { xs: 'w-3 h-3', sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-8 h-8' }

  return (
    <svg
      className={cn('animate-spin text-primary', sizes[size], className)}
      fill="none"
      viewBox="0 0 24 24"
      aria-label="Loading"
      role="status"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

/* ── Divider ────────────────────────────────────────────────── */

interface DividerProps {
  label?: string
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

export function Divider({ label, orientation = 'horizontal', className }: DividerProps) {
  if (orientation === 'vertical') {
    return <div className={cn('w-px self-stretch bg-border', className)} />
  }

  if (label) {
    return (
      <div className={cn('flex items-center gap-3', className)} role="separator">
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-muted-foreground whitespace-nowrap">{label}</span>
        <div className="flex-1 h-px bg-border" />
      </div>
    )
  }
  return <div className={cn('h-px bg-border', className)} role="separator" />
}

/* ── Card Deck ──────────────────────────────────────────────── */

interface CardDeckProps {
  title?: string
  description?: string
  footer?: ReactNode
  children: ReactNode
  className?: string
}

export function CardDeck({ title, description, footer, children, className }: CardDeckProps) {
  return (
    <div className={cn('border border-border rounded-lg overflow-hidden bg-card', className)}>
      {(title || description) && (
        <div className="px-4 py-3 border-b border-border">
          {title && <p className="font-medium text-sm text-foreground">{title}</p>}
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
      )}
      <div className="p-4">{children}</div>
      {footer && <div className="px-4 py-3 border-t border-border bg-muted/30">{footer}</div>}
    </div>
  )
}

/* ── Empty State ────────────────────────────────────────────── */

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
  size?: 'sm' | 'default' | 'lg'
  className?: string
}

export function EmptyState({ icon, title, description, action, size = 'default', className }: EmptyStateProps) {
  const paddings = { sm: 'py-8 px-3', default: 'py-12 px-4', lg: 'py-16 px-6' }

  return (
    <div className={cn('flex flex-col items-center justify-center text-center', paddings[size], className)}>
      {icon && (
        <div className="mb-4 flex items-center justify-center rounded-xl bg-muted/50 p-3 text-muted-foreground">
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description && <p className="text-xs text-muted-foreground mt-1 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

/* ── Search Field ───────────────────────────────────────────── */

interface SearchFieldProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  autoFocus?: boolean
}

export function SearchField({ value, onChange, placeholder = 'Search…', className, autoFocus }: SearchFieldProps) {
  return (
    <div className={cn('relative', className)}>
      <svg
        className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoFocus={autoFocus}
        className={cn(
          'w-full pl-8 pr-3 py-1.5 text-sm rounded-lg border border-input bg-background',
          'text-foreground placeholder:text-muted-foreground',
          'focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/60',
          'transition-[border-color,box-shadow] duration-200',
        )}
      />
      {value && (
        <button
          type="button"
          aria-label="Clear search"
          onClick={() => onChange('')}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground focus:outline-none"
        >
          <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}

/* ── Pagination ─────────────────────────────────────────────── */

interface PaginationProps {
  page: number
  total: number
  pageSize: number
  onChange: (page: number) => void
  className?: string
}

export function Pagination({ page, total, pageSize, onChange, className }: PaginationProps) {
  const totalPages = Math.ceil(total / pageSize)
  if (totalPages <= 1) return null

  const hasPrev = page > 1
  const hasNext = page < totalPages

  return (
    <div className={cn('flex items-center justify-between', className)}>
      <span className="text-xs text-muted-foreground">
        Page {page} of {totalPages} ({total} items)
      </span>
      <div className="flex gap-1">
        <button
          type="button"
          onClick={() => onChange(page - 1)}
          disabled={!hasPrev}
          aria-label="Previous page"
          className="px-2.5 py-1 text-xs rounded-md border border-border text-foreground hover:bg-muted/50 disabled:opacity-40 disabled:pointer-events-none transition-colors"
        >
          Prev
        </button>
        <button
          type="button"
          onClick={() => onChange(page + 1)}
          disabled={!hasNext}
          aria-label="Next page"
          className="px-2.5 py-1 text-xs rounded-md border border-border text-foreground hover:bg-muted/50 disabled:opacity-40 disabled:pointer-events-none transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  )
}
