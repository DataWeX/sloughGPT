import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface StatCardProps {
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
          <span className={cn("text-[10px]", trend.positive ? "text-green-500" : "text-red-500")}>
            {trend.positive ? '+' : ''}{trend.value}%
          </span>
        )}
      </div>
    </div>
  )
}
