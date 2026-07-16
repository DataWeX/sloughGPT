import { render, cleanup, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.hoisted(() => { (process.env as Record<string, string>).NODE_ENV = 'development' })

// Mock error-store
const mockAddError = vi.fn()
vi.mock('@/lib/error-store', () => ({
  useErrorStore: Object.assign(
    vi.fn((selector: any) => selector({ addError: mockAddError })),
    { getState: vi.fn(() => ({ addError: mockAddError })) },
  ),
}))

// Mock toast-store
const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(
    vi.fn((selector: any) => selector({ addToast: mockAddToast })),
    { getState: vi.fn(() => ({ addToast: mockAddToast })) },
  ),
}))

// Mock error-reporter
vi.mock('@/lib/error-reporter', () => ({
  reportError: vi.fn(),
  initErrorReporter: vi.fn(),
}))

// Mock dev-log
vi.mock('@/lib/dev-log', () => ({
  logger: { child: vi.fn(() => ({ info: vi.fn(), warning: vi.fn(), error: vi.fn() })) },
}))

import { ErrorLifecycle } from './ErrorLifecycle'

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('ErrorLifecycle', () => {
  it('renders without crashing', () => {
    const { container } = render(<ErrorLifecycle />)
    expect(container.innerHTML).toBe('')
  })

  it('installs window.error listener', () => {
    const spy = vi.spyOn(window, 'addEventListener')
    render(<ErrorLifecycle />)
    expect(spy).toHaveBeenCalledWith('error', expect.any(Function), true) // capture phase
    expect(spy).toHaveBeenCalledWith('error', expect.any(Function))      // bubble phase
    expect(spy).toHaveBeenCalledWith('unhandledrejection', expect.any(Function))
  })

  it('cleans up listeners on unmount', () => {
    const spy = vi.spyOn(window, 'removeEventListener')
    const { unmount } = render(<ErrorLifecycle />)
    unmount()
    expect(spy).toHaveBeenCalledWith('error', expect.any(Function), true)
    expect(spy).toHaveBeenCalledWith('error', expect.any(Function))
    expect(spy).toHaveBeenCalledWith('unhandledrejection', expect.any(Function))
  })

  it('does not double-init on re-render', () => {
    const spy = vi.spyOn(window, 'addEventListener')
    const { rerender } = render(<ErrorLifecycle />)
    rerender(<ErrorLifecycle />)
    // Should only have 3 addEventListener calls (error capture, error bubble, rejection)
    const errorCalls = spy.mock.calls.filter(c => c[0] === 'error')
    expect(errorCalls.length).toBe(2)
  })
})
