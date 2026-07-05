import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const { mockInit, mockReport } = vi.hoisted(() => ({
  mockInit: vi.fn(),
  mockReport: vi.fn(),
}))

vi.mock('@/lib/error-reporter', () => ({
  initErrorReporter: mockInit,
  reportError: mockReport,
}))

import { ErrorReporterInit } from './ErrorReporterInit'

describe('ErrorReporterInit', () => {
  let origConsoleError: typeof console.error

  beforeEach(() => {
    origConsoleError = console.error
    vi.clearAllMocks()
  })

  afterEach(() => {
    console.error = origConsoleError
    cleanup()
  })

  it('renders null', () => {
    const { container } = render(<ErrorReporterInit />)
    expect(container.innerHTML).toBe('')
  })

  it('calls initErrorReporter on mount', () => {
    render(<ErrorReporterInit />)
    expect(mockInit).toHaveBeenCalled()
  })

  it('reports hydration error and suppresses it', () => {
    render(<ErrorReporterInit />)
    console.error('Hydration failed because the client did not match')
    expect(mockReport).toHaveBeenCalled()
  })

  it('forwards non-hydration errors through patched console.error', () => {
    const callLog: unknown[][] = []
    console.error = (...args: unknown[]) => { callLog.push(args) }
    render(<ErrorReporterInit />)
    console.error('Some other error')
    expect(callLog.length).toBe(1)
    expect(callLog[0][0]).toBe('Some other error')
  })

  it('restores console.error on unmount', () => {
    const { unmount } = render(<ErrorReporterInit />)
    unmount()
    expect(console.error).toBe(origConsoleError)
  })
})
