import { describe, expect, it, vi, beforeEach } from 'vitest'
import { authController } from './auth-controller'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from './http-client'

describe('authController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('getMe calls apiGet with correct path and headers', async () => {
    const mockUser = { id: '1', username: 'test', email: 'test@example.com' }
    vi.mocked(apiGet).mockResolvedValue(mockUser)

    const result = await authController.getMe('my-token')

    expect(apiGet).toHaveBeenCalledWith('/auth/me', undefined, {
      headers: { Authorization: 'Bearer my-token' },
    })
    expect(result).toEqual(mockUser)
  })

  it('login calls apiPost with credentials', async () => {
    const mockResponse = { token: 'abc', user: { id: '1', username: 'u', email: 'e' } }
    vi.mocked(apiPost).mockResolvedValue(mockResponse)

    const result = await authController.login('user', 'pass')

    expect(apiPost).toHaveBeenCalledWith('/auth/login', { username: 'user', password: 'pass' })
    expect(result).toEqual(mockResponse)
  })

  it('register calls apiPost with all fields', async () => {
    const mockResponse = { token: 'xyz', user: { id: '2', username: 'new', email: 'n@e.com' } }
    vi.mocked(apiPost).mockResolvedValue(mockResponse)

    const result = await authController.register('new', 'n@e.com', 'pwd')

    expect(apiPost).toHaveBeenCalledWith('/auth/register', { username: 'new', email: 'n@e.com', password: 'pwd' })
    expect(result).toEqual(mockResponse)
  })

  it('verify calls apiPost with auth header', async () => {
    const mockResponse = { data: { valid: true } }
    vi.mocked(apiPost).mockResolvedValue(mockResponse)

    const result = await authController.verify('tok123')

    expect(apiPost).toHaveBeenCalledWith('/auth/verify', undefined, {
      headers: { Authorization: 'Bearer tok123' },
    })
    expect(result).toEqual(mockResponse)
  })

  it('getMe returns user data', async () => {
    vi.mocked(apiGet).mockResolvedValue({ id: '1', username: 'alice', email: 'a@b.com' })
    const result = await authController.getMe('token')
    expect(result.username).toBe('alice')
  })

  it('login returns token', async () => {
    vi.mocked(apiPost).mockResolvedValue({ token: 'xyz', user: { id: '1', username: 'u', email: 'e' } })
    const result = await authController.login('u', 'p')
    expect(result.token).toBe('xyz')
  })
})
