'use client'

import { type ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card'

export interface ActionCardProps {
  /** Card title */
  title: string
  /** Optional subtitle below the title */
  subtitle?: string
  /** Action buttons/elements displayed in the header */
  actions?: ReactNode
  /** Card body content */
  children: ReactNode
  /** Additional CSS classes for the outer Card */
  className?: string
  /** Additional CSS classes for the header */
  headerClassName?: string
  /** Additional CSS classes for the content */
  contentClassName?: string
  /** Test ID for testing */
  testId?: string
}

/**
 * Card with title, optional subtitle, and action buttons in the header.
 *
 * Common pattern for cards that have a title on the left and
 * action buttons on the right (e.g., refresh, settings, export).
 *
 * @example
 * ```tsx
 * <ActionCard
 *   title="System Health"
 *   subtitle="Last updated 2m ago"
 *   actions={
 *     <>
 *       <Button size="sm" variant="outline" onClick={onRefresh}>
 *         <IconRefresh className="h-3 w-3" />
 *       </Button>
 *       <Button size="sm" variant="ghost" onClick={onSettings}>
 *         <IconSettings className="h-3 w-3" />
 *       </Button>
 *     </>
 *   }
 * >
 *   <HealthMetrics />
 * </ActionCard>
 * ```
 */
export function ActionCard({
  title,
  subtitle,
  actions,
  children,
  className,
  headerClassName,
  contentClassName,
  testId,
}: ActionCardProps) {
  return (
    <Card className={className} data-testid={testId}>
      <CardHeader className={cn('flex flex-row items-center justify-between space-y-0', headerClassName)}>
        <div>
          <CardTitle className="text-base">{title}</CardTitle>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-1.5">
            {actions}
          </div>
        )}
      </CardHeader>
      <CardContent className={contentClassName}>
        {children}
      </CardContent>
    </Card>
  )
}
