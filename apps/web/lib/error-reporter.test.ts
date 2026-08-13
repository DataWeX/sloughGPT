// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const FETCH_URL = 'http://localhost:8000/errors/log'

const mockFetch = vi.fn().mockResolvedValue({ ok: true })
vi.stubGlobal('fetch', mockFetch)

vi.stubGlobal('window', {
  addEventListener: vi.fn(),
  location: { href: 'http://localhost:3000' },
  localStorage: { getItem: vi.fn(() => '[]'), setItem: vi.fn(), clear: vi.fn() },
})

import { reportError } from './error-reporter'

beforeEach(() => {
  vi.useFakeTimers()
  mockFetch.mockClear()
  // Reset module-level batch state by advancing timers so any pending flushes happen
  vi.advanceTimersByTime(6000)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('reportError', () => {
  it('flushes at MAX_BATCH_SIZE (10) — also tests batching', () => {
    for (let i = 0; i < 10; i++) reportError(`error ${i}`)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors).toHaveLength(10)
    expect(body.errors[0].message).toBe('error 0')
    expect(body.errors[9].message).toBe('error 9')
  })

  it('delays flush for partial batch', () => {
    reportError('single')
    expect(mockFetch).not.toHaveBeenCalled()
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors[0].message).toBe('single')
  })

  it('includes stack and extra metadata', () => {
    reportError('warn', 'test', { stack: 'line 1', metadata: { key: 'val' } })
    vi.advanceTimersByTime(5000)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors[0].message).toBe('warn')
    expect(body.errors[0].source).toBe('test')
    expect(body.errors[0].stack).toBe('line 1')
    expect(body.errors[0].metadata).toEqual({ key: 'val' })
  })

  it('reuses pending timer', () => {
    const spy = vi.spyOn(globalThis, 'setTimeout')
    reportError('first')
    reportError('second')
    expect(spy).toHaveBeenCalledTimes(1)
    spy.mockRestore()
  })

  it('handles fetch failure gracefully', async () => {
    mockFetch.mockRejectedValueOnce(new Error('network'))
    reportError('fail')
    vi.advanceTimersByTime(5000)
    // Should not throw
    expect(true).toBe(true)
  })

  it('batch array resets after flush', () => {
    for (let i = 0; i < 10; i++) reportError(`e${i}`)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    reportError('after-flush')
    expect(mockFetch).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })
})
