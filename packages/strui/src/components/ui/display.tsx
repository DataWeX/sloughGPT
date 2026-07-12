'use client'

import { forwardRef, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

/* ── Stat Card ──────────────────────────────────────────────── */

interface StatCardProps {
  label: string
  value: string | number | ReactNode
  icon?: ReactNode
  trend?: { value: number; positive?: boolean; label?: string }
  description?: string
  loading?: boolean
  className?: string
}

export function StatCard({ label, value, icon, trend, description, loading, className }: StatCardProps) {
  if (loading) {
    return (
      <div className={cn('flex flex-col p-3 rounded-lg border border-border/50 bg-card/50 gap-2', className)}>
        <div className="h-3 w-16 rounded bg-muted animate-pulse" />
        <div className="h-6 w-24 rounded bg-muted animate-pulse" />
      </div>
    )
  }

  return (
    <div className={cn('flex flex-col p-3 rounded-lg border border-border/50 bg-card/50 gap-1', className)}>
      <div className="flex items-center justify-between gap-1.5">
        <div className="flex items-center gap-1.5">
          {icon && <span className="text-muted-foreground">{icon}</span>}
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{label}</span>
        </div>
        {trend && (
          <span
            className={cn(
              'inline-flex items-center gap-0.5 text-[10px] font-semibold rounded-full px-1.5 py-0.5',
              trend.positive !== false
                ? 'text-success bg-success/10'
                : 'text-destructive bg-destructive/10',
            )}
          >
            {trend.positive !== false ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
        )}
      </div>
      <span className="text-xl font-bold tracking-tight text-foreground">{value}</span>
      {(description || trend?.label) && (
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          {trend?.label ?? description}
        </p>
      )}
    </div>
  )
}

/* ── KPI Grid ───────────────────────────────────────────────── */

interface KpiGridProps {
  children: ReactNode
  columns?: 1 | 2 | 3 | 4 | 5
  className?: string
}

export function KpiGrid({ children, columns = 4, className }: KpiGridProps) {
  const cols = {
    1: 'grid-cols-1',
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-3',
    4: 'grid-cols-2 sm:grid-cols-4',
    5: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5',
  }
  return (
    <div className={cn('grid gap-3', cols[columns], className)}>
      {children}
    </div>
  )
}

/* ── List Row ───────────────────────────────────────────────── */

interface ListRowProps {
  label: string
  value?: string | ReactNode
  action?: ReactNode
  icon?: ReactNode
  onClick?: () => void
  className?: string
}

export function ListRow({ label, value, action, icon, onClick, className }: ListRowProps) {
  const Tag = onClick ? 'button' : 'div'
  return (
    <Tag
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      className={cn(
        'flex w-full items-center justify-between py-2 px-3 text-left',
        'transition-colors duration-150',
        onClick && 'cursor-pointer hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
        !onClick && 'hover:bg-muted/30',
        className,
      )}
    >
      <div className="flex items-center gap-2 min-w-0">
        {icon && <span className="text-muted-foreground shrink-0">{icon}</span>}
        <span className="text-sm truncate">{label}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0 ml-3">
        {value && <span className="text-sm text-muted-foreground">{value}</span>}
        {action}
      </div>
    </Tag>
  )
}

/* ── List Section ───────────────────────────────────────────── */

interface ListSectionProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
}

export function ListSection({ title, description, children, className }: ListSectionProps) {
  return (
    <div className={cn('space-y-1', className)}>
      {(title || description) && (
        <div className="px-3 pb-1">
          {title && <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{title}</p>}
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
      )}
      <div className="border border-border rounded-lg overflow-hidden divide-y divide-border">{children}</div>
    </div>
  )
}

/* ── Empty Card ─────────────────────────────────────────────── */

interface EmptyCardProps {
  message?: string
  description?: string
  action?: ReactNode
  icon?: ReactNode
  className?: string
}

export function EmptyCard({ message = 'No items', description, action, icon, className }: EmptyCardProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12 px-4 text-center border border-dashed border-border rounded-lg', className)}>
      {icon && (
        <div className="mb-3 flex items-center justify-center w-10 h-10 rounded-xl bg-muted/50 text-muted-foreground">
          {icon}
        </div>
      )}
      <p className="text-sm font-medium text-foreground">{message}</p>
      {description && <p className="text-xs text-muted-foreground mt-1 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

/* ── Skeleton ───────────────────────────────────────────────── */

interface SkeletonProps {
  className?: string
  /** Number of lines to render (text skeleton) */
  lines?: number
  /** Width of last line as percentage (text skeleton) */
  lastLineWidth?: string
}

export function Skeleton({ className, lines, lastLineWidth = '60%' }: SkeletonProps) {
  if (lines && lines > 1) {
    return (
      <div className={cn('space-y-2', className)}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-4 animate-pulse rounded bg-muted"
            style={{ width: i === lines - 1 ? lastLineWidth : '100%' }}
          />
        ))}
      </div>
    )
  }
  return <div className={cn('animate-pulse rounded-md bg-muted', className)} />
}

/* ── Loading Dots ───────────────────────────────────────────── */

interface LoadingDotsProps {
  size?: 'sm' | 'default' | 'lg'
  className?: string
}

export function LoadingDots({ size = 'default', className }: LoadingDotsProps) {
  const dotSizes = { sm: 'w-1 h-1', default: 'w-1.5 h-1.5', lg: 'w-2 h-2' }
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn('inline-flex items-center gap-0.5', className)}
    >
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:0ms]')} />
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:150ms]')} />
      <span className={cn(dotSizes[size], 'bg-current rounded-full animate-bounce [animation-delay:300ms]')} />
    </span>
  )
}
