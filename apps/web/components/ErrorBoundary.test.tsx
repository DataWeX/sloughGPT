// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('@/lib/toast-store', () => ({
  useToastStore: { getState: () => ({ addToast: mockAddToast }) },
}))

vi.mock('@/lib/error-store', () => ({
  addGlobalError: vi.fn(),
}))

import { ErrorBoundary } from './ErrorBoundary'

function ThrowError() {
  throw new Error('Test error')
}

describe('ErrorBoundary', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders children when no error', () => {
    render(<ErrorBoundary><div>Normal content</div></ErrorBoundary>)
    expect(screen.getByText('Normal content')).toBeDefined()
  })

  it('shows error UI on caught error', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(<ErrorBoundary><ThrowError /></ErrorBoundary>)
    expect(screen.getByText('Something went wrong')).toBeDefined()
    expect(screen.getByText('Reload page')).toBeDefined()
    expect(screen.getByText('Go home')).toBeDefined()
  })

  it('renders custom fallback when provided', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(<ErrorBoundary fallback={<div>Custom Fallback</div>}><ThrowError /></ErrorBoundary>)
    expect(screen.getByText('Custom Fallback')).toBeDefined()
  })
})
