import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi, afterEach } from 'vitest'

import { NewDot } from './new-dot'

const FRESH = Date.parse('2026-09-25')
const STALE = Date.parse('2026-11-01')

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('NewDot', () => {
  it('renders pulsing dot for a fresh experimental feature', () => {
    vi.useFakeTimers()
    vi.setSystemTime(FRESH)
    const html = renderToStaticMarkup(<NewDot feature="consciousness" />)
    expect(html).toContain('aria-label="New"')
    expect(html).toContain('animate-ping')
  })

  it('renders nothing once the freshness window expires', () => {
    vi.useFakeTimers()
    vi.setSystemTime(STALE)
    expect(renderToStaticMarkup(<NewDot feature="consciousness" />)).toBe('')
  })

  it('renders nothing for unknown features', () => {
    expect(renderToStaticMarkup(<NewDot feature="nope" />)).toBe('')
  })

  it('renders nothing for non-experimental features', () => {
    expect(renderToStaticMarkup(<NewDot feature="chat" />)).toBe('')
  })

  it('uses a custom label', () => {
    vi.useFakeTimers()
    vi.setSystemTime(FRESH)
    const html = renderToStaticMarkup(<NewDot feature="voice" label="Beta" />)
    expect(html).toContain('aria-label="Beta"')
  })
})
