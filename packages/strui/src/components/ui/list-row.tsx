'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'

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
