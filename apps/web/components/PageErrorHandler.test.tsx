// @vitest-environment jsdom

import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { PageErrorHandler } from './PageErrorHandler'

vi.mock('@/lib/error-store', () => ({
  addGlobalError: vi.fn(),
}))

vi.mock('@/lib/error-reporter', () => ({
  reportError: vi.fn(),
}))

describe('PageErrorHandler', () => {
  const mockReset = vi.fn()
  const mockError = new Error('Test error message')

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders default title', () => {
    render(<PageErrorHandler error={mockError} reset={mockReset} />)
    expect(screen.getAllByText('Something went wrong').length).toBeGreaterThanOrEqual(1)
  })

  it('renders custom title', () => {
    render(<PageErrorHandler error={mockError} reset={mockReset} title="Custom Title" />)
    expect(screen.getAllByText('Custom Title').length).toBeGreaterThanOrEqual(1)
  })

  it('renders error message truncated', () => {
    render(<PageErrorHandler error={mockError} reset={mockReset} />)
    expect(screen.getAllByText('Test error message').length).toBeGreaterThanOrEqual(1)
  })

  it('calls reset on Try again click', () => {
    render(<PageErrorHandler error={mockError} reset={mockReset} />)
    fireEvent.click(screen.getAllByText('Try again')[0])
    expect(mockReset).toHaveBeenCalledOnce()
  })

  it('toggles details on Details click', () => {
    render(<PageErrorHandler error={mockError} reset={mockReset} />)
    const detailsBtn = screen.getAllByText('Details')[0]
    fireEvent.click(detailsBtn)
    expect(screen.getAllByText('Hide').length).toBeGreaterThanOrEqual(1)
    fireEvent.click(screen.getAllByText('Hide')[0])
    expect(screen.getAllByText('Details').length).toBeGreaterThanOrEqual(1)
  })

  it('reports error on mount', async () => {
    const { addGlobalError } = await import('@/lib/error-store')
    const { reportError } = await import('@/lib/error-reporter')
    render(<PageErrorHandler error={mockError} reset={mockReset} />)
    expect(addGlobalError).toHaveBeenCalledWith(mockError, 'PageErrorHandler')
    expect(reportError).toHaveBeenCalledWith(
      mockError.message,
      'page-error-boundary',
      expect.objectContaining({ stack: mockError.stack })
    )
  })
})
