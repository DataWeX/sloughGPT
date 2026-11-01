'use client'

import type { ReactNode } from 'react'

import { cn, EmptyCard } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { PageSkeleton } from '@/components/ui/PageSkeleton'
import { StatusBanner } from '@/components/composed/StatusBanner'

type MaxWidth = 'max-w-3xl' | 'max-w-4xl' | 'max-w-5xl' | 'max-w-6xl' | 'max-w-7xl' | 'max-w-none'

interface PageContainerProps {
  /** Page title — rendered inside AppRouteHeaderLead as h1. */
  title: ReactNode
  /** Optional subtitle below title. */
  subtitle?: ReactNode
  /** Right-side actions in the header. Wraps on mobile. */
  headerRight?: ReactNode
  /** Toolbar slot — search, filters, sort. Rendered between header and content. */
  toolbar?: ReactNode
  /** Max width of centered column. Default: max-w-4xl (896px). */
  maxWidth?: MaxWidth
  /** Additional classes on the outer sl-page wrapper. */
  className?: string
  /** Content classes applied to the inner content stack. */
  contentClassName?: string
  /** Show loading skeleton. */
  loading?: boolean
  /** Custom loading content — replaces default skeleton cards. */
  loadingContent?: ReactNode
  /** Number of skeleton cards while loading. */
  loadingCards?: number
  /** Show grid skeleton instead of card skeleton while loading. */
  loadingGrid?: boolean
  /** Error state — shows banner + retry. */
  error?: string | null
  /** Callback when retry button is clicked. */
  onRetry?: () => void
  /** Show empty state when true. Children are hidden. */
  empty?: boolean
  /** Empty state message. */
  emptyMessage?: string
  /** Empty state description. */
  emptyDescription?: string
  /** Empty state icon. */
  emptyIcon?: ReactNode
  /** Empty state action button. */
  emptyAction?: ReactNode
  /** Page content — hidden when loading, error, or empty. */
  children?: ReactNode
}

/**
 * Standard page container — single source of truth for all app pages.
 * Noir Violet design system — all colors via CSS variables, spacing via tokens.
 *
 * Structure:
 *   .sl-page (outer, handles responsive page padding + bg)
 *     └─ .mx-auto max-w-* w-full (inner, centered column)
 *          ├─ AppRouteHeader (title / subtitle / headerRight)
 *          ├─ toolbar (optional)
 *          └─ content | skeleton | error | empty
 *
 * Responsive:
 * - Padding via .sl-page: 12px → 16px → 24px → 32px (640/768/1024 breakpoints)
 * - Header: flex-wrap, gap-x-4, wraps on <sm
 * - Toolbar: full-width, min-h 44px touch targets
 * - Content: mt-6 space-y-4, animate-in fade-in (respects prefers-reduced-motion)
 * - No nested scroll — parent .sl-app-content handles overflow
 *
 * A11y:
 * - Title always h1 via AppRouteHeaderLead (sl-h1)
 * - Loading: aria-busy, Error: role=alert via StatusBanner, Empty: EmptyCard
 */
export function PageContainer({
  title,
  subtitle,
  headerRight,
  toolbar,
  maxWidth = 'max-w-4xl',
  className,
  contentClassName,
  loading = false,
  loadingContent,
  loadingCards = 3,
  loadingGrid = false,
  error = null,
  onRetry,
  empty = false,
  emptyMessage = 'Nothing here yet',
  emptyDescription,
  emptyIcon,
  emptyAction,
  children,
}: PageContainerProps) {
  const innerClass = cn('mx-auto w-full', maxWidth)

  // ── Loading ───────────────────────────────────────────────────────────────
  if (loading) {
    if (loadingContent) {
      return (
        <div className={cn('sl-page view-pane', className)} data-testid="page-container" data-state="loading">
          <div className={innerClass}>
            <AppRouteHeader
              left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
              right={headerRight}
            />
            {toolbar && <div className="mb-4 mt-4">{toolbar}</div>}
            <div className="mt-6" aria-busy="true" aria-live="polite">
              {loadingContent}
            </div>
          </div>
        </div>
      )
    }
    return (
      <div className={cn('sl-page view-pane', className)} data-testid="page-container" data-state="loading">
        <div className={innerClass}>
          <AppRouteHeader
            left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
            right={headerRight}
          />
          {toolbar && <div className="mb-4 mt-4">{toolbar}</div>}
          <div className="mt-6">
            <PageSkeleton cards={loadingCards} grid={loadingGrid} header={false} />
          </div>
        </div>
      </div>
    )
  }

  // ── Error ─────────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className={cn('sl-page view-pane', className)} data-testid="page-container" data-state="error">
        <div className={innerClass}>
          <AppRouteHeader
            left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
            right={headerRight}
          />
          {toolbar && <div className="mb-4 mt-4">{toolbar}</div>}
          <div className="mt-6 flex min-h-[40vh] flex-col items-center justify-center px-4 py-16 text-center sm:px-0">
            <div className="max-w-sm space-y-3" role="alert">
              <StatusBanner variant="error" message={error} dismissible={false} onRetry={onRetry} />
            </div>
          </div>
        </div>
      </div>
    )
  }

  // ── Empty ─────────────────────────────────────────────────────────────────
  if (empty) {
    return (
      <div className={cn('sl-page view-pane', className)} data-testid="page-container" data-state="empty">
        <div className={innerClass}>
          <AppRouteHeader
            left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
            right={headerRight}
          />
          {toolbar && <div className="mb-4 mt-4">{toolbar}</div>}
          <div className="mt-6 flex min-h-[40vh] flex-col items-center justify-center px-4 py-16 text-center sm:px-0">
            <EmptyCard
              message={emptyMessage}
              description={emptyDescription}
              icon={emptyIcon as any}
              action={(emptyAction ?? null) as any}
            />
          </div>
        </div>
      </div>
    )
  }

  // ── Default ───────────────────────────────────────────────────────────────
  return (
    <div className={cn('sl-page view-pane', className)} data-testid="page-container" data-state="ready">
      <div className={innerClass}>
        <AppRouteHeader
          left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
          right={headerRight}
        />
        {toolbar && <div className="mb-4 mt-4">{toolbar}</div>}
        <div className={cn('mt-6 space-y-4', contentClassName)}>{children}</div>
      </div>
    </div>
  )
}
