// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/react'
import React from 'react'

const { mockAddError, mockAddToast, mockUseErrorStore, mockUseToastStore } = vi.hoisted(() => ({
  mockAddError: vi.fn(),
  mockAddToast: vi.fn(),
  mockUseErrorStore: vi.fn(),
  mockUseToastStore: vi.fn(),
}))

vi.mock('@/lib/error-store', () => ({
  useErrorStore: mockUseErrorStore,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: mockUseToastStore,
}))

import { GlobalErrorHandler } from './GlobalErrorHandler'

function setupStoreMocks() {
  mockUseErrorStore.mockImplementation((selector: (s: any) => any) => {
    return selector({ addError: mockAddError, clearErrors: vi.fn() })
  })
  mockUseToastStore.mockImplementation((selector: (s: any) => any) => {
    return selector({ addToast: mockAddToast, clearToasts: vi.fn() })
  })
}

describe('GlobalErrorHandler', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupStoreMocks()
  })
  afterEach(cleanup)

  it('renders null', () => {
    const { container } = render(<GlobalErrorHandler />)
    expect(container.innerHTML).toBe('')
  })

  it('catches error events and calls addError/addToast', () => {
    render(<GlobalErrorHandler />)
    const event = new ErrorEvent('error', {
      message: 'runtime failure',
      filename: 'app.js',
      lineno: 42,
      colno: 10,
      error: new Error('runtime failure'),
    })
    window.dispatchEvent(event)
    expect(mockAddError).toHaveBeenCalled()
    expect(mockAddToast).toHaveBeenCalled()
  })

  it('shows toast for non-fatal errors (ResizeObserver)', () => {
    render(<GlobalErrorHandler />)
    const event = new ErrorEvent('error', {
      message: 'ResizeObserver loop limit exceeded',
      error: new Error('ResizeObserver loop limit exceeded'),
    })
    window.dispatchEvent(event)
    expect(mockAddError).not.toHaveBeenCalled()
    expect(mockAddToast).toHaveBeenCalledWith('ResizeObserver loop limit exceeded', 'info')
  })

  it('shows toast for hydration errors', () => {
    render(<GlobalErrorHandler />)
    const event = new ErrorEvent('error', {
      message: 'Hydration failed because the initial UI does not match',
      error: new Error('Hydration'),
    })
    window.dispatchEvent(event)
    expect(mockAddError).not.toHaveBeenCalled()
    expect(mockAddToast).toHaveBeenCalled()
  })

  it('cleans up event listeners on unmount', () => {
    const addErrorSpy = vi.spyOn(window, 'addEventListener')
    const removeSpy = vi.spyOn(window, 'removeEventListener')
    const { unmount } = render(<GlobalErrorHandler />)
    expect(addErrorSpy).toHaveBeenCalledWith('error', expect.any(Function))
    expect(addErrorSpy).toHaveBeenCalledWith('unhandledrejection', expect.any(Function))
    unmount()
    expect(removeSpy).toHaveBeenCalledWith('error', expect.any(Function))
    expect(removeSpy).toHaveBeenCalledWith('unhandledrejection', expect.any(Function))
    addErrorSpy.mockRestore()
    removeSpy.mockRestore()
  })
})
