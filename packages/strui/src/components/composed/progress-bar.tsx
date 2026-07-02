import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/cn'

export interface ProgressBarProps extends Omit<HTMLAttributes<HTMLDivElement>, 'children'> {
  /** 0–`max` (ignored when `indeterminate`). */
  value?: number
  max?: number
  /** Visually indeterminate (ignores value). */
  indeterminate?: boolean
}

/** Accessible linear progress for jobs, uploads, and context fill. */
export function ProgressBar({
  value = 0,
  max = 100,
  className,
  indeterminate,
  ...props
}: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      aria-valuetext={indeterminate ? undefined : `${Math.round(pct)}%`}
      className={cn('relative h-2 w-full overflow-hidden rounded-none bg-muted', className)}
      {...props}
    >
      {indeterminate ? (
        <div 
          className="absolute inset-0 h-full"
          style={{
            background: `linear-gradient(135deg, 
              #90EE90 25%, 
              transparent 25%, 
              transparent 50%, 
              #90EE90 50%, 
              #90EE90 75%, 
              transparent 75%
            )`,
            backgroundSize: '16px 16px',
            animation: 'progress-zigzag 0.8s linear infinite',
          }}
        />
      ) : (
        <>
          <div 
            className="absolute inset-0 h-full"
            style={{
              background: `linear-gradient(135deg, 
                rgba(144, 238, 144, 0.3) 25%, 
                transparent 25%, 
                transparent 50%, 
                rgba(144, 238, 144, 0.3) 50%, 
                rgba(144, 238, 144, 0.3) 75%, 
                transparent 75%
              )`,
              backgroundSize: '12px 12px',
              animation: 'progress-zigzag 0.6s linear infinite',
            }}
          />
          <div
            className="absolute inset-y-0 left-0 h-full transition-[width] duration-300 ease-smooth"
            style={{ 
              width: `${pct}%`,
              background: 'linear-gradient(90deg, #98FB98 0%, #7CFC00 50%, #98FB98 100%)',
            }}
          />
        </>
      )}
      <style>{`
        @keyframes progress-zigzag {
          0% { background-position: 0 0; }
          100% { background-position: 24px 0; }
        }
      `}</style>
    </div>
  )
}
