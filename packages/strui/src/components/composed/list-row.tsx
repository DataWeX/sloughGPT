import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

export interface ListRowProps {
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
