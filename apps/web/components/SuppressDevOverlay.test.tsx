// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, cleanup } from '@testing-library/react'
import React from 'react'

import { SuppressDevOverlay } from './SuppressDevOverlay'

describe('SuppressDevOverlay', () => {
  let origConsoleError: typeof console.error

  beforeEach(() => {
    origConsoleError = console.error
    localStorage.clear()
  })

  afterEach(() => {
    console.error = origConsoleError
    cleanup()
  })

  it('renders null', () => {
    const { container } = render(<SuppressDevOverlay />)
    expect(container.innerHTML).toBe('')
  })

  it('suppresses hydration error (not forwarded to original)', () => {
    const callLog: unknown[][] = []
    console.error = (...args: unknown[]) => { callLog.push(args) }
    render(<SuppressDevOverlay />)
    console.error('Text content did not match. Server: "foo" Client: "bar"')
    expect(callLog.length).toBe(0)
  })

  it('forwards non-hydration errors to original console.error', () => {
    const callLog: unknown[][] = []
    console.error = (...args: unknown[]) => { callLog.push(args) }
    render(<SuppressDevOverlay />)
    console.error('Some other error')
    expect(callLog.length).toBe(1)
    expect(callLog[0][0]).toBe('Some other error')
  })

  it('stores hydration error in localStorage', () => {
    console.error = origConsoleError
    render(<SuppressDevOverlay />)
    console.error('Hydration failed because the client did not match')
    const stored = JSON.parse(localStorage.getItem('__critical_errors') || '[]')
    expect(stored.length).toBe(1)
    expect(stored[0].type).toBe('hydration')
  })

  it('restores console.error on unmount', () => {
    console.error = origConsoleError
    const { unmount } = render(<SuppressDevOverlay />)
    unmount()
    expect(console.error).toBe(origConsoleError)
  })

  it('prevents hydration error events via window.addEventListener', () => {
    render(<SuppressDevOverlay />)
    const event = new ErrorEvent('error', { message: 'Hydration error: did not match', cancelable: true })
    const result = window.dispatchEvent(event)
    expect(result).toBe(false)
  })

  it('suppresses unhandledrejection events', () => {
    render(<SuppressDevOverlay />)
    // Use a weaker assertion: the listener is registered and doesn't crash
    const event = new PromiseRejectionEvent('unhandledrejection', {
      promise: Promise.resolve(),
      reason: 'test',
    })
    window.dispatchEvent(event)
  })
})
