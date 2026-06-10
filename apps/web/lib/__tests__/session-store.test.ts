import { describe, it, expect, beforeEach } from 'vitest'
import { sessionStore } from '../session-store'

describe('sessionStore', () => {
  beforeEach(() => {
    sessionStore.reset()
  })

  it('starts empty', () => {
    expect(sessionStore.isApproved('test-model')).toBe(false)
  })

  it('returns true after setApproved', () => {
    sessionStore.setApproved('gpt2')
    expect(sessionStore.isApproved('gpt2')).toBe(true)
  })

  it('returns false for unapproved models', () => {
    sessionStore.setApproved('gpt2')
    expect(sessionStore.isApproved('qwen')).toBe(false)
  })

  it('supports multiple models', () => {
    sessionStore.setApproved('a')
    sessionStore.setApproved('b')
    sessionStore.setApproved('c')
    expect(sessionStore.isApproved('a')).toBe(true)
    expect(sessionStore.isApproved('b')).toBe(true)
    expect(sessionStore.isApproved('c')).toBe(true)
    expect(sessionStore.isApproved('d')).toBe(false)
  })

  it('reset clears all approvals', () => {
    sessionStore.setApproved('gpt2')
    sessionStore.setApproved('qwen')
    sessionStore.reset()
    expect(sessionStore.isApproved('gpt2')).toBe(false)
    expect(sessionStore.isApproved('qwen')).toBe(false)
  })

  it('is isolated across resets', () => {
    sessionStore.setApproved('m1')
    sessionStore.reset()
    sessionStore.setApproved('m2')
    expect(sessionStore.isApproved('m1')).toBe(false)
    expect(sessionStore.isApproved('m2')).toBe(true)
  })
})
