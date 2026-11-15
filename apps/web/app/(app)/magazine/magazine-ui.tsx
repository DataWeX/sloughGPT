'use client'

/**
 * Shared presentational helpers for the brand guide: the color swatch and
 * the editorial section divider.
 */
import { cn } from '@sloughgpt/strui'

/* ──────────────────────────────────────────────────────────────────
 * Swatch — small reusable color swatch component
 * ────────────────────────────────────────────────────────────────── */

export function Swatch({
  rgb,
  label,
  description,
  size = 'md',
}: {
  rgb: string
  label: string
  description?: string
  size?: 'sm' | 'md' | 'lg'
}) {
  const sizeClasses = {
    sm: 'h-8 w-8 rounded',
    md: 'h-12 w-12 rounded-md',
    lg: 'h-16 w-16 rounded-lg',
  }
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className={cn(
          sizeClasses[size],
          'border border-border/40 shadow-sm transition-transform hover:scale-110',
        )}
        style={{ backgroundColor: `rgb(${rgb})` }}
        role="img"
        aria-label={`${label}: rgb(${rgb})`}
      />
      <div className="text-center">
        <div className="text-[10px] font-medium">{label}</div>
        <div className="font-mono text-[10px] text-muted-foreground">{rgb}</div>
        {description && <div className="text-[10px] text-muted-foreground">{description}</div>}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────
 * SectionHeader — editorial section divider
 * ────────────────────────────────────────────────────────────────── */

export function SectionHeader({ number, title }: { number: string; title: string }) {
  return (
    <div className="flex items-baseline gap-3 pt-2">
      <span className="font-mono text-xs text-muted-foreground">{number}</span>
      <h2 className="text-[11px] font-medium">{title}</h2>
      <div className="h-px flex-1 bg-border" />
    </div>
  )
}
