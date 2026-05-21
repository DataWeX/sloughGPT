'use client'

import { forwardRef, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface StatCardProps {
  label: string
  value: string | number
  icon?: ReactNode
  trend?: { value: number; positive?: boolean }
  className?: string
}

export function StatCard({ label, value, icon, trend, className }: StatCardProps) {
  return (
    <div className={cn("flex flex-col p-3 rounded-lg border border-border/50 bg-card/50", className)}>
      <div className="flex items-center gap-1.5 mb-1">
        {icon && <span className="text-muted-foreground">{icon}</span>}
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-xl font-semibold">{value}</span>
        {trend && (
          <span className={cn("text-[10px]", trend.positive ? "text-success" : "text-destructive")}>
            {trend.positive ? '+' : ''}{trend.value}%
          </span>
        )}
      </div>
    </div>
  )
}

interface KpiGridProps {
  children: ReactNode
  columns?: 2 | 3 | 4
  className?: string
}

export function KpiGrid({ children, columns = 4, className }: KpiGridProps) {
  return (
    <div className={cn(
      "grid gap-3",
      columns === 2 && "grid-cols-2",
      columns === 3 && "grid-cols-3",
      columns === 4 && "grid-cols-2 sm:grid-cols-4",
      className
    )}>
      {children}
    </div>
  )
}

interface ListRowProps {
  label: string
  value?: string
  action?: ReactNode
  className?: string
}

export function ListRow({ label, value, action, className }: ListRowProps) {
  return (
    <div className={cn(
      "flex items-center justify-between py-2 px-3 rounded-md hover:bg-muted/30 transition-colors",
      className
    )}>
      <span className="text-sm">{label}</span>
      {value && <span className="text-sm text-muted-foreground">{value}</span>}
      {action}
    </div>
  )
}

interface ListSectionProps {
  title?: string
  children: ReactNode
  className?: string
}

export function ListSection({ title, children, className }: ListSectionProps) {
  return (
    <div className={cn("space-y-1", className)}>
      {title && <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider px-3">{title}</p>}
      <div className="border rounded-lg overflow-hidden">{children}</div>
    </div>
  )
}

interface EmptyCardProps {
  message?: string
  action?: ReactNode
  className?: string
}

export function EmptyCard({ message = 'No items', action, className }: EmptyCardProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center py-12 px-4 text-center border border-dashed rounded-lg", className)}>
      <p className="text-sm text-muted-foreground">{message}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

interface SkeletonProps {
  className?: string
}

export function Skeleton({ className }: SkeletonProps) {
  return <div className={cn("animate-pulse rounded-md bg-muted", className)} />
}

interface LoadingDotsProps {
  className?: string
}

export function LoadingDots({ className }: LoadingDotsProps) {
  return (
    <div className={cn("flex items-center gap-0.5", className)}>
      <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:0ms]" />
      <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:150ms]" />
      <span className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:300ms]" />
    </div>
  )
}