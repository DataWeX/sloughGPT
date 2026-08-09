import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const h = vi.hoisted(() => ({
  onReport: undefined as undefined | ((metric: any) => void),
  logWarning: vi.fn(),
}))

vi.mock('next/web-vitals', () => ({
  useReportWebVitals: (cb: any) => { h.onReport = cb },
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ warning: h.logWarning }) },
}))

import WebVitals from './WebVitals'

function report(name: string, value: number, rating: string) {
  h.onReport!({ name, value, rating })
}

describe('WebVitals', () => {
  beforeEach(() => {
    h.logWarning.mockClear()
    h.onReport = undefined
  })

  it('registers the web vitals reporter and renders null', () => {
    const { container } = render(<WebVitals />)
    expect(h.onReport).toBeDefined()
    expect(container.firstChild).toBeNull()
    cleanup()
  })

  it('logs a fast metric as not slow', () => {
    render(<WebVitals />)
    report('CLS', 0.02, 'good')
    expect(h.logWarning).toHaveBeenCalledWith('web-vitals CLS', {
      value: 0,
      rating: 'good',
      slow: false,
    })
    cleanup()
  })

  it('flags a metric above its threshold as slow', () => {
    render(<WebVitals />)
    report('LCP', 3000, 'poor')
    expect(h.logWarning).toHaveBeenCalledWith('web-vitals LCP', {
      value: 3000,
      rating: 'poor',
      slow: true,
    })
    cleanup()
  })

  it('never flags a metric without a threshold', () => {
    render(<WebVitals />)
    report('UNKNOWN_METRIC', 999999, 'good')
    expect(h.logWarning).toHaveBeenCalledWith('web-vitals UNKNOWN_METRIC', {
      value: 999999,
      rating: 'good',
      slow: false,
    })
    cleanup()
  })

  it('rounds the value to an integer', () => {
    render(<WebVitals />)
    report('TTFB', 812.7, 'needs-improvement')
    expect(h.logWarning).toHaveBeenCalledWith('web-vitals TTFB', {
      value: 813,
      rating: 'needs-improvement',
      slow: true,
    })
    cleanup()
  })

  it('handles multiple reports in a session', () => {
    render(<WebVitals />)
    report('CLS', 0.05, 'needs-improvement')
    report('INP', 100, 'good')
    expect(h.logWarning).toHaveBeenCalledTimes(2)
    cleanup()
  })
})
