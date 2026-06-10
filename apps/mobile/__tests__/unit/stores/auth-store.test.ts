import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useAuthStore } from '@/stores/auth-store'

describe('Auth Store', () => {
  beforeEach(() => {
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    })
  })

  it('should initialize with default state', () => {
    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(state.isAuthenticated).toBe(false)
    expect(state.isLoading).toBe(false)
  })

  it('should login successfully', () => {
    const user = { id: '1', username: 'testuser', email: 'test@example.com' }
    const token = 'test-token-123'

    useAuthStore.getState().login(user, token)

    const state = useAuthStore.getState()
    expect(state.user).toEqual(user)
    expect(state.token).toBe(token)
    expect(state.isAuthenticated).toBe(true)
  })

  it('should logout successfully', () => {
    // First login
    useAuthStore.getState().login(
      { id: '1', username: 'testuser', email: 'test@example.com' },
      'test-token'
    )

    // Then logout
    useAuthStore.getState().logout()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(state.isAuthenticated).toBe(false)
  })

  it('should set user', () => {
    const user = { id: '2', username: 'newuser', email: 'new@example.com' }
    useAuthStore.getState().setUser(user)

    expect(useAuthStore.getState().user).toEqual(user)
  })

  it('should set token and update isAuthenticated', () => {
    useAuthStore.getState().setToken('new-token')
    expect(useAuthStore.getState().token).toBe('new-token')
    expect(useAuthStore.getState().isAuthenticated).toBe(true)

    useAuthStore.getState().setToken(null)
    expect(useAuthStore.getState().token).toBeNull()
    expect(useAuthStore.getState().isAuthenticated).toBe(false)
  })

  it('should set loading state', () => {
    useAuthStore.getState().setLoading(true)
    expect(useAuthStore.getState().isLoading).toBe(true)

    useAuthStore.getState().setLoading(false)
    expect(useAuthStore.getState().isLoading).toBe(false)
  })
})
