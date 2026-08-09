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
    expect(errors.length).toBe(32)
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
    expect(loadings.length).toBe(32)
  })

  it.each(loadings)('renders %s with a skeleton', (key) => {
    const Component = fallbacks[key].default
    const { container } = render(<Component />)
    expect(container.querySelector('.animate-pulse, .animate-spin')).toBeTruthy()
  })
})
