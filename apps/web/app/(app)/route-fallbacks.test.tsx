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
    expect(errors.length).toBeGreaterThanOrEqual(26)
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
    expect(loadings.length).toBeGreaterThanOrEqual(24)
  })

  it.each(loadings)('renders %s with a skeleton', (key) => {
    const Component = fallbacks[key].default
    const { container } = render(<Component />)
    expect(container.querySelector('.animate-pulse, .animate-spin')).toBeTruthy()
  })
})
