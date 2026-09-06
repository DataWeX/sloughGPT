'use client'

import type { ReactNode } from 'react'

import { cn, Button, IconRefresh, EmptyCard } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { PageSkeleton } from '@/components/ui/PageSkeleton'

interface PageContainerProps {
  /** Page title — rendered inside AppRouteHeaderLead. */
  title: ReactNode
  /** Optional subtitle below title. */
  subtitle?: string
  /** Right-side actions in the header. Wraps on mobile. */
  headerRight?: ReactNode
  /** Toolbar slot — search bar, filter buttons, sort controls. Rendered between header and content. */
  toolbar?: ReactNode
  /** Max width of the page container. Default: max-w-4xl (896px). */
  maxWidth?: 'max-w-4xl' | 'max-w-5xl' | 'max-w-6xl' | 'max-w-7xl'
  /** Additional classes on the outer sl-page wrapper. */
  className?: string
  /** Content classes applied to the inner scroll container. */
  contentClassName?: string
  /** Show full-page loading skeleton. */
  loading?: boolean
  /** Custom loading content — replaces default PageSkeleton when provided. */
  loadingContent?: ReactNode
  /** Number of skeleton cards while loading (used only when loadingContent is not provided). */
  loadingCards?: number
  /** Show grid skeleton instead of card skeleton while loading. */
  loadingGrid?: boolean
  /** Error state — shows error message + retry button. */
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
  children: ReactNode
}

/**
 * Standard page container with header, loading, error, empty, and content states.
 *
 * Responsive behavior:
 * - Padding: 12px → 16px → 24px → 32px (via sl-page)
 * - Title: text-2xl → text-3xl (via sl-h1)
 * - Header: flex-wrap, gap shrinks on mobile
 * - Toolbar: full-width on mobile, inline on desktop
 * - Content: space-y-4, scrolls independently
 * - Error/Empty: centered on all breakpoints, min-h-[40vh]
 * - Touch targets: minimum 44px (h-11) on actionable elements
 *
 * @example Simple page
 * ```tsx
 * <PageContainer title="Settings" loading={isLoading}>
 *   <Card>...</Card>
 * </PageContainer>
 * ```
 *
 * @example Page with toolbar and empty state
 * ```tsx
 * <PageContainer
 *   title="Knowledge"
 *   subtitle="Manage facts the AI remembers"
 *   headerRight={<Button>Add fact</Button>}
 *   toolbar={<Input placeholder="Search..." />}
 *   loading={loading}
 *   error={error}
 *   onRetry={refetch}
 *   empty={items.length === 0}
 *   emptyMessage="No knowledge stored"
 *   emptyDescription="Add facts the AI should remember."
 *   emptyIcon={<IconSearch />}
 *   emptyAction={<Button>Add fact</Button>}
 * >
 *   <ItemList items={items} />
 * </PageContainer>
 * ```
 *
 * @example Custom loading state
 * ```tsx
 * <PageContainer
 *   title="Knowledge"
 *   loading={loading}
 *   loadingContent={<ListSkeleton items={5} />}
 * >
 *   ...
 * </PageContainer>
 * ```
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
  const wrapperClass = cn('sl-page mx-auto', maxWidth, className)
  if (loading) {
    if (loadingContent) {
      return (
        <div className={wrapperClass}>
          <AppRouteHeader
            left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
            right={headerRight}
          />
          {loadingContent}
        </div>
      )
    }
    return (
      <div className={wrapperClass}>
        <PageSkeleton cards={loadingCards} header grid={loadingGrid} />
      </div>
    )
  }

  if (error) {
    return (
      <div className={wrapperClass}>
        <AppRouteHeader
          left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
          right={headerRight}
        />
        <div className="flex min-h-[40vh] flex-col items-center justify-center px-4 py-16 text-center sm:px-0">
          <div className="max-w-sm space-y-3">
            <p className="text-sm text-destructive">{error}</p>
            {onRetry && (
              <Button size="sm" variant="outline" className="h-11 min-w-[120px]" onClick={onRetry}>
                <IconRefresh className="h-4 w-4 mr-1.5" />
                Retry
              </Button>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (empty) {
    return (
      <div className={wrapperClass}>
        <AppRouteHeader
          left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
          right={headerRight}
        />
        {toolbar && <div className="mb-4">{toolbar}</div>}
        <div className="flex min-h-[40vh] flex-col items-center justify-center px-4 py-16 text-center sm:px-0">
          <EmptyCard
            message={emptyMessage}
            description={emptyDescription}
            icon={emptyIcon as any}
            action={(emptyAction ?? null) as any}
          />
        </div>
      </div>
    )
  }

  return (
    <div className={wrapperClass}>
      <AppRouteHeader
        left={<AppRouteHeaderLead title={title} subtitle={subtitle} />}
        right={headerRight}
      />
      {toolbar && <div className="mb-4">{toolbar}</div>}
      <div className={cn('space-y-4', contentClassName)}>
        {children}
      </div>
    </div>
  )
}
