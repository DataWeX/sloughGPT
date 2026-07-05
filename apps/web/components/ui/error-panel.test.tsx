/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const mockErrors = [
  { id: 'e1', title: 'Server Error', message: '500 internal error', severity: 'error' as const, source: 'api', timestamp: Date.now(), dismissible: true },
  { id: 'e2', title: 'Not Found', message: 'resource missing', severity: 'warning' as const, timestamp: Date.now(), dismissible: true },
  { id: 'e3', title: 'Info', message: 'background sync', severity: 'info' as const, timestamp: Date.now(), dismissible: true },
]

let currentState = { errors: mockErrors, dismissError: vi.fn(), clearErrors: vi.fn() }

vi.mock('@/lib/error-store', () => ({
  useErrorStore: (selector: (s: Record<string, unknown>) => unknown) => selector(currentState as unknown as Record<string, unknown>),
}))

import { ErrorPanel } from './error-panel'

afterEach(cleanup)

beforeEach(() => {
  Object.assign(navigator, {
    clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
  })
  currentState = { errors: mockErrors, dismissError: vi.fn(), clearErrors: vi.fn() }
})

describe('ErrorPanel', () => {
  it('returns null when there are no errors', () => {
    currentState.errors = []
    const { container } = render(<ErrorPanel />)
    expect(container.innerHTML).toBe('')
  })

  it('renders error count button', () => {
    render(<ErrorPanel />)
    expect(screen.getByText('3 issues')).toBeInTheDocument()
  })

  it('renders error severity text', () => {
    render(<ErrorPanel />)
    expect(screen.getByText('3 issues · 1 error')).toBeInTheDocument()
  })

  it('renders error item title', () => {
    render(<ErrorPanel />)
    expect(screen.getByText('Server Error')).toBeInTheDocument()
  })

  it('renders error item message', () => {
    render(<ErrorPanel />)
    expect(screen.getByText('500 internal error')).toBeInTheDocument()
  })

  it('renders source when provided', () => {
    render(<ErrorPanel />)
    expect(screen.getByText(/Source: api/)).toBeInTheDocument()
  })

  it('renders copy button on each error item', () => {
    render(<ErrorPanel />)
    const copyBtns = screen.getAllByRole('button', { name: /copy error/i })
    expect(copyBtns).toHaveLength(3)
  })

  it('copies error details on copy button click', () => {
    render(<ErrorPanel />)
    const copyBtn = screen.getAllByRole('button', { name: /copy error/i })[0]
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
    const text = vi.mocked(navigator.clipboard.writeText).mock.calls[0][0]
    expect(text).toContain('Server Error')
    expect(text).toContain('500 internal error')
  })

  it('dismisses error when dismiss button clicked', () => {
    render(<ErrorPanel />)
    const dismissBtns = screen.getAllByRole('button', { name: 'Dismiss' })
    expect(dismissBtns).toHaveLength(3)
    fireEvent.click(dismissBtns[0])
    expect(currentState.dismissError).toHaveBeenCalledWith('e1')
  })

  it('shows copy all and clear all when multiple errors', () => {
    render(<ErrorPanel />)
    expect(screen.getByText('Copy all')).toBeInTheDocument()
    expect(screen.getByText('Clear all')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Clear all'))
    expect(currentState.clearErrors).toHaveBeenCalledOnce()
  })

  it('copies all errors when copy all clicked', () => {
    render(<ErrorPanel />)
    fireEvent.click(screen.getByText('Copy all'))
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
    const text = vi.mocked(navigator.clipboard.writeText).mock.calls[0][0]
    expect(text).toContain('Server Error')
    expect(text).toContain('Not Found')
    expect(text).toContain('background sync')
  })

  it('hides copy all and clear all for single error', () => {
    currentState.errors = [mockErrors[0]]
    render(<ErrorPanel />)
    expect(screen.queryByText('Copy all')).not.toBeInTheDocument()
    expect(screen.queryByText('Clear all')).not.toBeInTheDocument()
    expect(screen.getByText('1 issue')).toBeInTheDocument()
  })

  it('renders warning severity', () => {
    currentState.errors = [mockErrors[1]]
    render(<ErrorPanel />)
    expect(screen.getByText('Not Found')).toBeInTheDocument()
  })

  it('renders info severity', () => {
    currentState.errors = [mockErrors[2]]
    render(<ErrorPanel />)
    expect(screen.getByText('background sync')).toBeInTheDocument()
  })
})
