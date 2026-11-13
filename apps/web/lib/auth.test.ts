import { beforeEach, describe, expect, it } from 'vitest'
import { useAuthStore } from './auth'

describe('useAuthStore', () => {
  beforeEach(() => {
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      workspaces: [],
      currentWorkspace: null,
    })
  })

  it('starts unauthenticated', () => {
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('login sets user and token', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.user?.username).toBe('alice')
    expect(state.token).toBe('abc123')
  })

  it('logout clears state', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    useAuthStore.getState().logout()
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('setUser updates user only', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    useAuthStore.getState().setUser({ id: '1', username: 'bob', email: 'b@c.com', role: 'admin', status: 'active', tenant_id: 't1' })
    const state = useAuthStore.getState()
    expect(state.user?.username).toBe('bob')
    expect(state.token).toBe('abc123')
  })

  it('setUser(null) clears user but keeps token', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    useAuthStore.getState().setUser(null)
    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBe('abc123')
  })

  it('login then logout then login works', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    useAuthStore.getState().logout()
    useAuthStore.getState().login(
      { id: '2', username: 'bob', email: 'b@c.com', role: 'user', status: 'active', tenant_id: 't1' },
      'xyz789'
    )
    const state = useAuthStore.getState()
    expect(state.user?.username).toBe('bob')
    expect(state.token).toBe('xyz789')
  })

  it('setWorkspaces stores workspaces and auto-selects first', () => {
    const workspaces = [
      { id: 'ws1', name: 'Workspace 1', tenant_id: 't1', description: '', role: 'admin' },
      { id: 'ws2', name: 'Workspace 2', tenant_id: 't1', description: '', role: 'member' },
    ]
    useAuthStore.getState().setWorkspaces(workspaces)
    const state = useAuthStore.getState()
    expect(state.workspaces).toHaveLength(2)
    expect(state.currentWorkspace?.id).toBe('ws1')
  })

  it('switchWorkspace changes current workspace', () => {
    const workspaces = [
      { id: 'ws1', name: 'Workspace 1', tenant_id: 't1', description: '', role: 'admin' },
      { id: 'ws2', name: 'Workspace 2', tenant_id: 't1', description: '', role: 'member' },
    ]
    useAuthStore.getState().setWorkspaces(workspaces)
    useAuthStore.getState().switchWorkspace('ws2')
    const state = useAuthStore.getState()
    expect(state.currentWorkspace?.id).toBe('ws2')
  })

  it('logout clears workspaces', () => {
    useAuthStore.getState().login(
      { id: '1', username: 'alice', email: 'a@b.com', role: 'user', status: 'active', tenant_id: 't1' },
      'abc123'
    )
    useAuthStore.getState().setWorkspaces([
      { id: 'ws1', name: 'Workspace 1', tenant_id: 't1', description: '', role: 'admin' },
    ])
    useAuthStore.getState().logout()
    const state = useAuthStore.getState()
    expect(state.workspaces).toHaveLength(0)
    expect(state.currentWorkspace).toBeNull()
  })
})
