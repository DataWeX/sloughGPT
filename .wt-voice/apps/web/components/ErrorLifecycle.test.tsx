import { render, cleanup, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.hoisted(() => {
  ;(process.env as Record<string, string>).NODE_ENV = 'development'
})

// Mock error-store
const mockAddError = vi.fn()
vi.mock('@/lib/error-store', async (importOriginal) => {
  const actual = await (importOriginal as unknown as () => Promise<Record<string, unknown>>)()
  return {
    ...actual,
    useErrorStore: Object.assign(
      vi.fn((selector: any) => selector({ addError: mockAddError })),
      { getState: vi.fn(() => ({ addError: mockAddError })) },
    ),
  }
})

// Mock state-events (structured logging bridge)
vi.mock('@/lib/state-events', () => ({
  logStateEvent: vi.fn(),
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
const mockReportError = vi.fn()
vi.mock('@/lib/error-reporter', () => ({
  reportError: (...args: unknown[]) => mockReportError(...args),
  initErrorReporter: vi.fn(),
}))

// Mock chatDB (hydration path persists to Dexie)
vi.mock('@/lib/db', () => ({
  chatDB: { addError: vi.fn(() => Promise.resolve()) },
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
    expect(spy).toHaveBeenCalledWith('error', expect.any(Function)) // bubble phase
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
    const errorCalls = spy.mock.calls.filter((c) => c[0] === 'error')
    expect(errorCalls.length).toBe(2)
  })

  it('captures unhandled rejection events', () => {
    const spy = vi.spyOn(window, 'addEventListener')
    render(<ErrorLifecycle />)
    const rejectionCalls = spy.mock.calls.filter((c) => c[0] === 'unhandledrejection')
    expect(rejectionCalls.length).toBeGreaterThanOrEqual(1)
  })

  it('installs listeners in capture and bubble phases', () => {
    const spy = vi.spyOn(window, 'addEventListener')
    render(<ErrorLifecycle />)
    const errorCalls = spy.mock.calls.filter((c) => c[0] === 'error')
    expect(errorCalls.length).toBe(2)
    expect(errorCalls[0][2]).toBe(true)
    expect(errorCalls[1][2]).toBeUndefined()
  })

  it('treats minified React #418 as hydration — no fatal toast', () => {
    render(<ErrorLifecycle />)
    const msg =
      'Error: Minified React error #418; visit https://react.dev/errors/418?args[]=HTML&args[]= for the full message'
    const event = new ErrorEvent('error', { message: msg, cancelable: true })
    act(() => {
      window.dispatchEvent(event)
    })
    expect(event.defaultPrevented).toBe(true)
    const fatalCalls = mockAddToast.mock.calls.filter((c) => c[0] === 'Something went wrong.')
    expect(fatalCalls).toHaveLength(0)
    expect(mockReportError).toHaveBeenCalledWith(
      expect.stringContaining('#418'),
      'hydration',
      expect.objectContaining({ metadata: { minified: true } }),
    )
    expect(mockAddError).not.toHaveBeenCalled()
  })

  it('still fatals on a real runtime error', () => {
    render(<ErrorLifecycle />)
    const event = new ErrorEvent('error', {
      message: 'TypeError: Cannot read properties of undefined',
      cancelable: true,
    })
    act(() => {
      window.dispatchEvent(event)
    })
    expect(mockAddToast).toHaveBeenCalledWith('Something went wrong.', 'error', undefined)
    expect(mockAddError).toHaveBeenCalled()
  })

  it('classifies #423 and #425 minified hydration codes', () => {
    render(<ErrorLifecycle />)
    for (const code of [423, 425]) {
      mockAddToast.mockClear()
      mockReportError.mockClear()
      const event = new ErrorEvent('error', {
        message: `Minified React error #${code}; visit https://react.dev/errors/${code}`,
        cancelable: true,
      })
      act(() => {
        window.dispatchEvent(event)
      })
      expect(mockReportError).toHaveBeenCalledWith(
        expect.stringContaining(`#${code}`),
        'hydration',
        expect.anything(),
      )
      const fatalCalls = mockAddToast.mock.calls.filter((c) => c[0] === 'Something went wrong.')
      expect(fatalCalls).toHaveLength(0)
    }
  })
})
