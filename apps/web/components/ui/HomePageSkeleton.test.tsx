import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
  Card: ({ children, ...p }: any) => <div {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import {
  ActiveModelBannerSkeleton,
  StatsGridSkeleton,
  FeedbackBarSkeleton,
  TrainingStatusSkeleton,
  QuickActionsSkeleton,
  RecentActivitySkeleton,
  UsageStatsSkeleton,
  SystemHealthSkeleton,
  NavigationGridSkeleton,
} from './HomePageSkeleton'

const SKELETONS = [
  ActiveModelBannerSkeleton,
  StatsGridSkeleton,
  FeedbackBarSkeleton,
  TrainingStatusSkeleton,
  QuickActionsSkeleton,
  RecentActivitySkeleton,
  UsageStatsSkeleton,
  SystemHealthSkeleton,
  NavigationGridSkeleton,
]

describe('HomePageSkeleton', () => {
  it.each(SKELETONS)('%p renders without crashing', (Skeleton) => {
    const { container } = render(<Skeleton />)
    expect(container.firstChild).toBeTruthy()
    expect(container.querySelectorAll('[data-testid="skeleton"]').length).toBeGreaterThan(0)
  })
})
