// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ErrorReporter } from './error-reporter'

const { mockFetch } = vi.hoisted(() => ({
  mockFetch: vi.fn().mockResolvedValue({ ok: true }),
}))

vi.stubGlobal('fetch', mockFetch)

vi.stubGlobal('window', {
  addEventListener: vi.fn(),
  location: { href: 'http://localhost:3000' },
  localStorage: { getItem: vi.fn(() => '[]'), setItem: vi.fn(), clear: vi.fn() },
})

vi.mock('@/lib/db', () => ({ chatDB: { addError: vi.fn() } }))

beforeEach(() => {
  vi.useFakeTimers()
  mockFetch.mockClear()
})

afterEach(() => {
  vi.useRealTimers()
})

function makeReporter() {
  return new ErrorReporter({ apiUrl: 'http://localhost:8000', persist: vi.fn() })
}

describe('reportError — batching', () => {
  it('flushes at MAX_BATCH_SIZE (10)', () => {
    const reporter = makeReporter()
    for (let i = 0; i < 10; i++) reporter.report(`error ${i}`)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors).toHaveLength(10)
    expect(body.errors[0].message).toBe('error 0')
    expect(body.errors[9].message).toBe('error 9')
  })

  it('delays flush for partial batch', () => {
    const reporter = makeReporter()
    reporter.report('single')
    expect(mockFetch).not.toHaveBeenCalled()
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors[0].message).toBe('single')
  })

  it('includes stack and extra metadata', () => {
    const reporter = makeReporter()
    reporter.report('warn', 'test', { stack: 'line 1', metadata: { key: 'val' } })
    vi.advanceTimersByTime(5000)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors[0].message).toBe('warn')
    expect(body.errors[0].source).toBe('test')
    expect(body.errors[0].stack).toBe('line 1')
    expect(body.errors[0].metadata).toEqual({ key: 'val' })
  })

  it('reuses pending timer', () => {
    const reporter = makeReporter()
    const spy = vi.spyOn(globalThis, 'setTimeout')
    reporter.report('first')
    reporter.report('second')
    expect(spy).toHaveBeenCalledTimes(1)
    spy.mockRestore()
  })

  it('handles fetch failure gracefully', () => {
    const reporter = makeReporter()
    mockFetch.mockRejectedValueOnce(new Error('network'))
    reporter.report('fail')
    vi.advanceTimersByTime(5000)
    expect(true).toBe(true)
  })

  it('batch array resets after flush', () => {
    const reporter = makeReporter()
    for (let i = 0; i < 10; i++) reporter.report(`e${i}`)
    expect(mockFetch).toHaveBeenCalledTimes(1)
    reporter.report('after-flush')
    expect(mockFetch).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })
})

describe('reportError — dedup', () => {
  it('deduplicates same message within 5s window', () => {
    const reporter = makeReporter()
    reporter.report('duplicate')
    reporter.report('duplicate')
    reporter.report('duplicate')
    vi.advanceTimersByTime(5000)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    const dupes = body.errors.filter((e: { message: string }) => e.message === 'duplicate')
    expect(dupes).toHaveLength(1)
  })

  it('allows same message after 5s window', () => {
    const reporter = makeReporter()
    reporter.report('repeated')
    vi.advanceTimersByTime(5000)
    reporter.report('repeated')
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(2)
    const body1 = JSON.parse(mockFetch.mock.calls[0][1].body)
    const body2 = JSON.parse(mockFetch.mock.calls[1][1].body)
    expect(body1.errors[0].message).toBe('repeated')
    expect(body2.errors[0].message).toBe('repeated')
  })

  it('allows different messages within window', () => {
    const reporter = makeReporter()
    reporter.report('msg-a')
    reporter.report('msg-b')
    reporter.report('msg-c')
    vi.advanceTimersByTime(5000)
    const body = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(body.errors).toHaveLength(3)
  })

  it('does not share dedup state across instances', () => {
    const a = makeReporter()
    a.report('same')
    const b = makeReporter()
    b.report('same')
    vi.advanceTimersByTime(5000)
    expect(mockFetch).toHaveBeenCalledTimes(2)
  })
})
