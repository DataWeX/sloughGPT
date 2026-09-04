'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Card, CardContent } from '../ui/card'

export interface InfoCardProps {
  /** Icon displayed above the title */
  icon?: ReactNode
  /** Card title */
  title: string
  /** Optional description below the title */
  description?: string
  /** Card content */
  children?: ReactNode
  /** Icon tone (default: primary) */
  tone?: 'primary' | 'muted' | 'success' | 'warning' | 'destructive'
  /** Card size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
  /** Test ID for testing */
  testId?: string
}

const toneClasses = {
  primary: 'text-primary',
  muted: 'text-muted-foreground',
  success: 'text-green-500',
  warning: 'text-yellow-500',
  destructive: 'text-destructive',
}

const sizeClasses = {
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
}

/**
 * Card with icon, title, and optional description.
 *
 * Common pattern for feature cards, capability cards, and
 * status overview cards that lead with an icon.
 *
 * @example
 * ```tsx
 * <InfoCard
 *   icon={<IconCpu className="h-5 w-5" />}
 *   title="GPU Acceleration"
 *   description="CUDA cores available"
 *   tone="success"
 * >
 *   <p className="text-xs">Device: NVIDIA RTX 4090</p>
 * </InfoCard>
 * ```
 */
export function InfoCard({
  icon,
  title,
  description,
  children,
  tone = 'primary',
  size = 'md',
  className,
  testId,
}: InfoCardProps) {
  return (
    <Card className={cn(sizeClasses[size], className)} data-testid={testId}>
      <div className="flex items-start gap-3">
        {icon && (
          <div className={cn('mt-0.5 shrink-0', toneClasses[tone])}>
            {icon}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium">{title}</h3>
          {description && (
            <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
          )}
          {children && <div className="mt-2">{children}</div>}
        </div>
      </div>
    </Card>
  )
}
