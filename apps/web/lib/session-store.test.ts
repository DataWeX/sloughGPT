// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { sessionStore } from './session-store'

describe('sessionStore', () => {
  afterEach(() => { sessionStore.reset() })

  it('isApproved returns false for unknown key', () => {
    expect(sessionStore.isApproved('model-x')).toBe(false)
  })

  it('isApproved returns true after setApproved', () => {
    sessionStore.setApproved('model-x')
    expect(sessionStore.isApproved('model-x')).toBe(true)
  })

  it('tracks multiple keys independently', () => {
    sessionStore.setApproved('a')
    sessionStore.setApproved('b')
    expect(sessionStore.isApproved('a')).toBe(true)
    expect(sessionStore.isApproved('b')).toBe(true)
    expect(sessionStore.isApproved('c')).toBe(false)
  })

  it('reset clears all approved keys', () => {
    sessionStore.setApproved('model-x')
    sessionStore.reset()
    expect(sessionStore.isApproved('model-x')).toBe(false)
  })
})
