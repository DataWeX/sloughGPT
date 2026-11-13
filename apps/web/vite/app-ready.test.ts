import { describe, expect, it } from 'vitest'

import {
  APP_READY_ATTR,
  isStartupComplete,
  markAppReady,
  type AppReadyTarget,
} from './app-ready'

type Fake = AppReadyTarget & { attrs: Map<string, string> }

function fakeTarget(): Fake {
  const attrs = new Map<string, string>()
  return {
    attrs,
    hasAttribute: (name: string) => attrs.has(name),
    setAttribute: (name: string, value: string) => {
      attrs.set(name, value)
    },
  }
}

describe('markAppReady', () => {
  it('declares data-app-ready exactly once; later calls are no-ops', () => {
    const target = fakeTarget()

    expect(markAppReady(target)).toBe(true)
    expect(target.attrs.get(APP_READY_ATTR)).toBe('true')

    // Route navigations mount the sentinel again — set-once, first caller wins.
    expect(markAppReady(target)).toBe(false)
    expect(target.attrs.get(APP_READY_ATTR)).toBe('true')
  })

  it('is null-safe (no root element, no crash, no flag)', () => {
    expect(markAppReady(null)).toBe(false)
    expect(markAppReady(undefined)).toBe(false)
  })
})

describe('isStartupComplete', () => {
  it('accepts the two stages the StartupOverlay treats as booted', () => {
    expect(isStartupComplete('ready')).toBe(true)
    expect(isStartupComplete('background')).toBe(true)
  })

  it('rejects everything still booting or unknown', () => {
    expect(isStartupComplete('init')).toBe(false)
    expect(isStartupComplete('critical')).toBe(false)
    expect(isStartupComplete('unknown')).toBe(false)
    expect(isStartupComplete(null)).toBe(false)
    expect(isStartupComplete(undefined)).toBe(false)
  })
})
