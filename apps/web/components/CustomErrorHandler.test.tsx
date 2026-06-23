// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockAddGlobalError, mockReportError } = vi.hoisted(() => ({
  mockAddGlobalError: vi.fn(),
  mockReportError: vi.fn(),
}))

vi.mock('@/lib/error-store', () => ({
  addGlobalError: mockAddGlobalError,
}))

vi.mock('@/lib/error-reporter', () => ({
  reportError: mockReportError,
}))

import { CustomErrorHandler } from './CustomErrorHandler'

function createError(message: string, name = 'Error'): Error & { digest?: string } {
  const e = new Error(message)
  e.name = name
  return e
}

describe('CustomErrorHandler', () => {
  const reset = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders error message', () => {
    render(<CustomErrorHandler error={createError('test error')} reset={reset} />)
    expect(screen.getByText('test error')).toBeDefined()
  })

  it('shows "Something went wrong" for generic errors', () => {
    render(<CustomErrorHandler error={createError('test error')} reset={reset} />)
    expect(screen.getByText('Something went wrong')).toBeDefined()
  })

  it('shows "Connection Error" for network errors', () => {
    render(<CustomErrorHandler error={createError('Failed to fetch')} reset={reset} />)
    expect(screen.getByText('Connection Error')).toBeDefined()
  })

  it('shows "Authentication Error" for 401', () => {
    render(<CustomErrorHandler error={createError('401 Unauthorized')} reset={reset} />)
    expect(screen.getByText('Authentication Error')).toBeDefined()
  })

  it('shows "Page Not Found" for 404', () => {
    render(<CustomErrorHandler error={createError('404 Not Found')} reset={reset} />)
    expect(screen.getByText('Page Not Found')).toBeDefined()
  })

  it('toggles detail section', () => {
    render(<CustomErrorHandler error={createError('test')} reset={reset} />)
    fireEvent.click(screen.getByText('Details'))
    expect(screen.getByText('Error Details')).toBeDefined()
    fireEvent.click(screen.getByText('Hide'))
    expect(screen.queryByText('Error Details')).toBeNull()
  })

  it('calls reset on Try again', () => {
    render(<CustomErrorHandler error={createError('test')} reset={reset} />)
    fireEvent.click(screen.getByText('Try again'))
    expect(reset).toHaveBeenCalled()
  })

  it('calls addGlobalError and reportError on mount', () => {
    render(<CustomErrorHandler error={createError('mount test')} reset={reset} />)
    expect(mockAddGlobalError).toHaveBeenCalledWith(expect.any(Error), 'CustomErrorHandler')
    expect(mockReportError).toHaveBeenCalledWith('mount test', 'error-boundary', expect.any(Object))
  })

  it('shows network hint for fetch errors', () => {
    render(<CustomErrorHandler error={createError('Failed to fetch')} reset={reset} />)
    expect(screen.getByText(/Check your internet connection/)).toBeDefined()
  })

  it('copies error details to clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    render(<CustomErrorHandler error={createError('test')} reset={reset} />)
    fireEvent.click(screen.getByText('Details'))
    fireEvent.click(screen.getByText('Copy'))
    await waitFor(() => { expect(writeText).toHaveBeenCalled() })
  })
})
