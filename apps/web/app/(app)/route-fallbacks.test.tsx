/// <reference types="vite/client" />
import type { ComponentType } from 'react'
import { render, fireEvent, cleanup, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('@/lib/error-store', () => ({
  addGlobalError: vi.fn(),
}))

vi.mock('@/lib/error-reporter', () => ({
  reportError: vi.fn(),
}))

const fallbacks = import.meta.glob('./**/{error,loading}.tsx', {
  eager: true,
}) as Record<string, { default: ComponentType<any> }>

describe('route error boundaries', () => {
  const errors = Object.keys(fallbacks).filter((key) => key.endsWith('error.tsx'))

  afterEach(() => cleanup())

  it('covers every route error boundary', () => {
    // Route inventory churns as routes are added/removed — assert a sane
    // lower bound here; the it.each below verifies every boundary renders.
    // 2026-10-24: floor 50 -> 26 — the 36 shadowed pages (and their
    // error/loading companions) were deleted, page.tsx 109 -> 73.
    // 2026-11-07: floor 26 -> 25 — d0a57fba5 composed /model/[id] into
    // /models (route map card 3/7) and deleted its error/loading/page
    // trio; page-baseline.json was updated in the same commit, this
    // companion floor was missed.
    // 2026-11-08: floor 25 -> 24 — /vm folded into /developer as its VM tab
    // (route map card 6/7) and deleted its error/loading/page trio.
    // (route map card 7/7 deleted knowledge/[id]/page.tsx WITHOUT moving
    // this floor — that dir had no error.tsx companion to lose.)
    expect(errors.length).toBeGreaterThanOrEqual(24)
  })

  it.each(errors)('renders %s and retries', (key) => {
    const Component = fallbacks[key].default
    const reset = vi.fn()
    const { container } = render(<Component error={new Error('boom')} reset={reset} />)
    fireEvent.click(within(container).getByText('Try again'))
    expect(reset).toHaveBeenCalledOnce()
  })
})

describe('route loading skeletons', () => {
  const loadings = Object.keys(fallbacks).filter((key) => key.endsWith('loading.tsx'))

  afterEach(() => cleanup())

  it('covers every route loading fallback', () => {
    // See note above — update if routes change.
    // 2026-10-24: floor 48 -> 24 — same page-prune as the error boundary
    // (two kept routes carry an error.tsx but no loading.tsx).
    // 2026-11-07: floor 24 -> 23 — same /model/[id] deletion as above.
    // 2026-11-08: floor 23 -> 22 — same /vm deletion as above (card 6/7).
    // (card 7/7's knowledge/[id] deletion carried no loading.tsx — floor
    // stays 22, see the errors floor comment above.)
    expect(loadings.length).toBeGreaterThanOrEqual(22)
  })

  it.each(loadings)('renders %s with a skeleton', (key) => {
    const Component = fallbacks[key].default
    const { container } = render(<Component />)
    expect(container.querySelector('.animate-pulse, .animate-spin')).toBeTruthy()
  })
})
