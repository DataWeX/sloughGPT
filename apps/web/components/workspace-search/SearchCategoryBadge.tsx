'use client'

import { Badge } from '@sloughgpt/strui'

interface SearchCategoryBadgeProps {
  type: string
  label: string
  colorClass?: string
  count?: number
}

const DEFAULT_COLORS: Record<string, string> = {
  member: 'bg-primary/15 text-primary',
  training: 'bg-info/15 text-info',
  dataset: 'bg-success/15 text-success',
  knowledge: 'bg-accent text-accent',
}

export function SearchCategoryBadge({ type, label, colorClass, count }: SearchCategoryBadgeProps) {
  const color = colorClass ?? DEFAULT_COLORS[type] ?? 'bg-muted text-muted-foreground'

  return (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium ${color}`}
    >
      {label}
      {count !== undefined && <span className="text-muted-foreground">({count})</span>}
    </span>
  )
}
