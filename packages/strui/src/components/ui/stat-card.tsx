'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface StatCardProps {
  label: string
  value: string | number | ReactNode
  icon?: ReactNode
  trend?: { value: number; positive?: boolean; label?: string }
  description?: string
  loading?: boolean
  className?: string
  numeric?: boolean
}

export function StatCard({ label, value, icon, trend, description, loading, className, numeric }: StatCardProps) {
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
      <span className={cn("text-xl font-bold tracking-tight text-foreground", numeric && "font-numeric")}>{value}</span>
      {(description || trend?.label) && (
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          {trend?.label ?? description}
        </p>
      )}
    </div>
  )
}
