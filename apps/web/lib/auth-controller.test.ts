import { beforeEach, describe, expect, it, vi } from 'vitest'

// Mock localStorage for node environment
function mockStorage(): Storage {
  let store: Record<string, string> = {}
  return {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => { store[k] = v },
    removeItem: (k: string) => { delete store[k] },
    clear: () => { store = {} },
    get length() { return Object.keys(store).length },
    key: (i: number) => Object.keys(store)[i] ?? null,
  }
}
const storage = mockStorage()
vi.stubGlobal('localStorage', storage)

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { authController } from './auth-controller'

describe('authController.login', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    localStorage.clear()
  })

  it('POSTs to /auth/login with credentials', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ token: 'abc123', user: { id: 'u1', username: 'test', email: 'test@x.com' } }),
    } as Response)

    const result = await authController.login('test', 'pass')
    expect(result.token).toBe('abc123')
    expect(result.user.username).toBe('test')
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/auth/login'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ username: 'test', password: 'pass' }),
      }),
    )
  })

  it('throws on failed login', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'Invalid credentials' }),
    } as Response)

    await expect(authController.login('test', 'wrong')).rejects.toThrow('Invalid credentials')
  })

  it('throws generic error when no detail', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => { throw new Error('parse error') },
    } as unknown as Response)

    await expect(authController.login('test', 'wrong')).rejects.toThrow('Login failed')
  })
})

describe('authController.register', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    localStorage.clear()
  })

  it('POSTs to /auth/register', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ token: 'xyz', user: { id: 'u2', username: 'newuser', email: 'new@x.com' } }),
    } as Response)

    const result = await authController.register('newuser', 'new@x.com', 'pass')
    expect(result.token).toBe('xyz')
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/auth/register'),
      expect.objectContaining({ method: 'POST' }),
    )
  })
})

describe('authController.logout', () => {
  beforeEach(() => { localStorage.clear() })

  it('clears localStorage tokens', () => {
    localStorage.setItem('auth_token', 'abc')
    localStorage.setItem('auth_user', 'test')

    authController.logout()

    expect(localStorage.getItem('auth_token')).toBeNull()
    expect(localStorage.getItem('auth_user')).toBeNull()
  })
})

describe('authController.getToken / setToken', () => {
  beforeEach(() => { localStorage.clear() })

  it('returns null when no token', () => {
    expect(authController.getToken()).toBeNull()
  })

  it('stores and retrieves token', () => {
    authController.setToken('mytoken')
    expect(authController.getToken()).toBe('mytoken')
  })
})
