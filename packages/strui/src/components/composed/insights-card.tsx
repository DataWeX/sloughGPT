'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card'
import { KpiGrid, type KpiGridProps } from './kpi-grid'
import { StatCard, type StatCardProps } from './stat-card'

export interface InsightsCardDetail {
  label: string
  value: ReactNode
  /** Optional className for the value */
  valueClassName?: string
}

export interface InsightsCardProps {
  /** Card title */
  title: string
  /** KPI stats to display in the grid */
  kpis?: StatCardProps[]
  /** Number of KPI columns (default: auto based on kpis length) */
  kpiColumns?: KpiGridProps['columns']
  /** Detail rows below the KPI grid */
  details?: InsightsCardDetail[]
  /** Optional extra content below details */
  children?: ReactNode
  /** Additional CSS classes for the outer Card */
  className?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Insights card with KPI grid and detail rows.
 *
 * Combines a title, optional KPI stats, and a list of label/value
 * detail rows. Used for domain-specific insight panels (feedback,
 * learning, datasets, benchmarks, etc.).
 *
 * @example
 * ```tsx
 * <InsightsCard
 *   title="Feedback Insights"
 *   kpis={[
 *     { label: 'Sentiment', value: '85%' },
 *     { label: 'Quality', value: 'Excellent' },
 *   ]}
 *   details={[
 *     { label: 'Total conversations', value: 42 },
 *     { label: 'Total messages', value: 128 },
 *   ]}
 * />
 * ```
 */
export function InsightsCard({
  title,
  kpis = [],
  kpiColumns,
  details = [],
  children,
  className,
  testId,
}: InsightsCardProps) {
  const autoColumns = kpiColumns ?? (kpis.length <= 2 ? 2 : kpis.length <= 4 ? 4 : 3)

  return (
    <Card className={className} data-testid={testId}>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {kpis.length > 0 && (
          <KpiGrid columns={autoColumns} className="mb-3">
            {kpis.map((kpi, i) => (
              <StatCard key={i} {...kpi} />
            ))}
          </KpiGrid>
        )}

        {details.length > 0 && (
          <div className="space-y-1.5 text-[11px]">
            {details.map((detail, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-muted-foreground">{detail.label}</span>
                <span className={cn('font-numeric', detail.valueClassName)}>
                  {detail.value}
                </span>
              </div>
            ))}
          </div>
        )}

        {children}
      </CardContent>
    </Card>
  )
}
