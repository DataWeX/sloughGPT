import { beforeEach, describe, expect, it } from 'vitest'
import { useAuthStore } from './auth'

describe('useAuthStore', () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null, token: null, isAuthenticated: false })
  })

  it('starts unauthenticated', () => {
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('login sets user and token', () => {
    useAuthStore.getState().login({ id: '1', username: 'alice', email: 'a@b.com' }, 'abc123')
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.user?.username).toBe('alice')
    expect(state.token).toBe('abc123')
  })

  it('logout clears state', () => {
    useAuthStore.getState().login({ id: '1', username: 'alice', email: 'a@b.com' }, 'abc123')
    useAuthStore.getState().logout()
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('setUser updates user only', () => {
    useAuthStore.getState().login({ id: '1', username: 'alice', email: 'a@b.com' }, 'abc123')
    useAuthStore.getState().setUser({ id: '1', username: 'bob', email: 'b@c.com' })
    const state = useAuthStore.getState()
    expect(state.user?.username).toBe('bob')
    expect(state.token).toBe('abc123')
  })

  it('setUser(null) clears user but keeps token', () => {
    useAuthStore.getState().login({ id: '1', username: 'alice', email: 'a@b.com' }, 'abc123')
    useAuthStore.getState().setUser(null)
    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBe('abc123')
  })

  it('login then logout then login works', () => {
    useAuthStore.getState().login({ id: '1', username: 'alice', email: 'a@b.com' }, 'abc123')
    useAuthStore.getState().logout()
    useAuthStore.getState().login({ id: '2', username: 'bob', email: 'b@c.com' }, 'xyz789')
    const state = useAuthStore.getState()
    expect(state.user?.username).toBe('bob')
    expect(state.token).toBe('xyz789')
  })
})
