'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Card, CardContent } from '../ui/card'
import { KpiGrid, type KpiGridProps } from './kpi-grid'

export interface MetricsCardProps {
  /** Section title displayed above the KPI grid */
  title: string
  /** KPI grid columns (default: 2) */
  columns?: KpiGridProps['columns']
  /** Content to render inside the card */
  children: ReactNode
  /** Additional CSS classes for the outer Card */
  className?: string
  /** Additional CSS classes for the title */
  titleClassName?: string
  /** Additional CSS classes for the CardContent */
  contentClassName?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Dashboard section card with a section title and content area.
 *
 * Commonly used in monitoring dashboards to group related KPIs
 * under a labeled section. Wraps content in a Card with consistent
 * padding and typography.
 *
 * @example
 * ```tsx
 * <MetricsCard title="Resources" columns={2}>
 *   <StatCard label="CPU" value="45%" />
 *   <StatCard label="Memory" value="8.2 GB" />
 * </MetricsCard>
 * ```
 */
export function MetricsCard({
  title,
  columns = 2,
  children,
  className,
  titleClassName,
  contentClassName,
  testId,
}: MetricsCardProps) {
  return (
    <Card className={cn('p-3', className)} data-testid={testId}>
      <span
        className={cn(
          'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block',
          titleClassName
        )}
      >
        {title}
      </span>
      <CardContent className={cn('p-0', contentClassName)}>
        <KpiGrid columns={columns}>
          {children}
        </KpiGrid>
      </CardContent>
    </Card>
  )
}
