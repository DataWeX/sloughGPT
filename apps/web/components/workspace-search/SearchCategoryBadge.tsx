'use client'

import { Badge } from '@sloughgpt/strui'

interface SearchCategoryBadgeProps {
  type: string
  label: string
  colorClass?: string
  count?: number
}

const DEFAULT_COLORS: Record<string, string> = {
  member: 'bg-purple-100 text-purple-700',
  training: 'bg-blue-100 text-blue-700',
  dataset: 'bg-green-100 text-green-700',
  knowledge: 'bg-orange-100 text-orange-700',
}

export function SearchCategoryBadge({
  type,
  label,
  colorClass,
  count,
}: SearchCategoryBadgeProps) {
  const color = colorClass ?? DEFAULT_COLORS[type] ?? 'bg-muted text-muted-foreground'

  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium ${color}`}>
      {label}
      {count !== undefined && (
        <span className="text-muted-foreground">({count})</span>
      )}
    </span>
  )
}
