'use client'

/**
 * Miniature trait visualizations shared by the Souls tabs and dialogs:
 * an SVG radar polygon and a labelled progress bar. Pure render.
 */
import { cn } from '@sloughgpt/strui'
import { traitColor } from './souls-format'

// ── Trait Radar (mini visualization) ──

export function TraitRadar({
  values,
  size = 80,
}: {
  values: Record<string, number>
  size?: number
}) {
  const entries = Object.entries(values).slice(0, 8)
  if (entries.length === 0) return null
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 4
  const angleStep = (2 * Math.PI) / entries.length

  const points = entries.map(([_, v], i) => {
    const angle = i * angleStep - Math.PI / 2
    const dist = r * Math.max(0.1, Math.min(1, v))
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) }
  })

  // <polygon points> takes an SVG point list ("x,y x,y …"), not path syntax
  // ("M…L…Z") — path commands make the browser reject the attribute.
  const polygonPoints = points.map((p) => `${p.x},${p.y}`).join(' ')

  return (
    <svg
      width={size}
      height={size}
      className="shrink-0"
      role="img"
      aria-label="Soul trait radar chart"
    >
      {[0.25, 0.5, 0.75, 1].map((scale) => (
        <polygon
          key={scale}
          points={entries
            .map((_, i) => {
              const angle = i * angleStep - Math.PI / 2
              return `${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`
            })
            .join(' ')}
          fill="none"
          stroke="currentColor"
          className="text-border/40"
          strokeWidth={0.5}
        />
      ))}
      <polygon
        points={polygonPoints}
        fill="rgb(var(--primary))"
        fillOpacity={0.15}
        stroke="rgb(var(--primary))"
        strokeWidth={1}
      />
    </svg>
  )
}

// ── Personality Bar ──

export function PersonalityBar({
  label,
  value,
  color = 'bg-primary',
}: {
  label: string
  value: number
  color?: string
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-0.5">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn('font-mono font-medium', traitColor(value))}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', color)}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
    </div>
  )
}
