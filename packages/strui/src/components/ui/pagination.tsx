'use client'

import { cn } from '../../lib/cn'

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
