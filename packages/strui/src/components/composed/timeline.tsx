import type { ReactNode } from 'react'

import { cn } from '../../lib/cn'

import type { StatusDotTone } from './status-dot'
import { STATUS_DOT_TONE_CLASSES } from './status-dot'

export interface TimelineItem {
  id: string
  title: ReactNode
  /** Secondary line (timestamp, subtitle). */
  meta?: ReactNode
  /** Dot color; defaults follow index (last = primary). */
  tone?: StatusDotTone
}

export interface TimelineProps {
  items: TimelineItem[]
  className?: string
}

const defaultTones: StatusDotTone[] = ['muted', 'muted', 'primary', 'success']

/** Vertical step / event list for training phases, deploy stages, audit trails. */
export function Timeline({ items, className }: TimelineProps) {
  return (
    <ul className={cn('relative space-y-0', className)} role="list">
      {items.map((item, index) => {
        const tone = item.tone ?? defaultTones[Math.min(index, defaultTones.length - 1)]
        const isLast = index === items.length - 1
        return (
          <li key={item.id} className="relative flex gap-3 pb-6 last:pb-0">
            {!isLast ? (
              <span
                className="absolute left-[4px] top-4 h-[calc(100%-0.25rem)] w-px bg-border"
                aria-hidden
              />
            ) : null}
            <div className="relative z-[1] mt-0.5 shrink-0 rounded-full bg-card p-px ring-1 ring-border">
              <span
                className={cn(
                  'block h-2 w-2 rounded-full',
                  STATUS_DOT_TONE_CLASSES[tone],
                )}
                aria-hidden
              />
            </div>
            <div className="min-w-0 pt-0.5">
              <p className="text-sm font-medium text-foreground">{item.title}</p>
              {item.meta ? (
                <div className="mt-0.5 text-xs text-muted-foreground">{item.meta}</div>
              ) : null}
            </div>
          </li>
        )
      })}
    </ul>
  )
}
