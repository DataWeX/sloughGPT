'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Chip, type ChipProps } from './chip'

export interface ChipGroupProps {
  /** Array of chip configurations */
  chips: (ChipProps & { key?: string })[]
  /** Spacing between chips (default: 1) */
  gap?: 0 | 1 | 2 | 3
  /** Additional CSS classes for the container */
  className?: string
  /** Test ID for testing */
  testId?: string
}

const gapClasses = {
  0: 'gap-0',
  1: 'gap-1',
  2: 'gap-2',
  3: 'gap-3',
}

/**
 * Group of chips/tags with consistent spacing.
 *
 * Renders a flex container with chips. Useful for displaying
 * tags, categories, status indicators, or filter chips.
 *
 * @example
 * ```tsx
 * <ChipGroup
 *   chips={[
 *     { label: 'Python', tone: 'primary' },
 *     { label: 'PyTorch', tone: 'success' },
 *     { label: 'GPU', tone: 'warning' },
 *   ]}
 * />
 * ```
 */
export function ChipGroup({
  chips,
  gap = 1,
  className,
  testId,
}: ChipGroupProps) {
  return (
    <div
      className={cn('flex flex-wrap items-center', gapClasses[gap], className)}
      data-testid={testId}
    >
      {chips.map((chip, i) => {
        const { key, ...chipProps } = chip
        return (
          <Chip key={key ?? i} {...chipProps} />
        )
      })}
    </div>
  )
}
