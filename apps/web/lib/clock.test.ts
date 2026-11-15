import { afterEach, describe, expect, it } from 'vitest'
import {
  getServerClockOffsetMs,
  resetServerClock,
  sampleServerTime,
  serverNow,
} from '../../lib/clock'

afterEach(() => {
  resetServerClock()
})

describe('server clock', () => {
  it('trusts the local clock before any sample (offset 0)', () => {
    expect(getServerClockOffsetMs()).toBe(0)
    const before = Date.now()
    const t = serverNow()
    expect(t).toBeGreaterThanOrEqual(before)
    expect(t).toBeLessThanOrEqual(Date.now())
  })

  it('applies a +90s skew from the Date response header', () => {
    const local = Date.now()
    const headers = new Headers({ date: new Date(local + 90_000).toUTCString() })
    sampleServerTime(headers)
    // HTTP Date has 1s granularity: allow quantization, demand the skew.
    expect(serverNow()).toBeGreaterThanOrEqual(local + 89_000)
    expect(serverNow()).toBeLessThan(local + 91_000)
  })

  it('applies a -90s skew (server behind the client)', () => {
    const local = Date.now()
    const headers = new Headers({ date: new Date(local - 90_000).toUTCString() })
    sampleServerTime(headers)
    expect(serverNow()).toBeLessThanOrEqual(local - 89_000)
    expect(serverNow()).toBeGreaterThan(local - 91_000)
  })

  it('last sample wins (offset tracks the newest response)', () => {
    const local = Date.now()
    sampleServerTime(new Headers({ date: new Date(local + 90_000).toUTCString() }))
    sampleServerTime(new Headers({ date: new Date(local).toUTCString() }))
    expect(Math.abs(getServerClockOffsetMs())).toBeLessThan(1_000)
  })

  it('ignores a missing Date header (keeps the previous offset)', () => {
    const local = Date.now()
    sampleServerTime(new Headers({ date: new Date(local + 90_000).toUTCString() }))
    const before = getServerClockOffsetMs()
    sampleServerTime(new Headers())
    expect(getServerClockOffsetMs()).toBe(before)
  })

  it('ignores an unparseable Date header', () => {
    sampleServerTime(new Headers({ date: 'not-a-date' }))
    expect(getServerClockOffsetMs()).toBe(0)
  })
})
