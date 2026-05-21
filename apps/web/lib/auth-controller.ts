/**
 * Auth Controller — axios-based API for authentication.
 *
 * Uses raw fetch for login/register to avoid auth interceptor (no token yet).
 */

import { apiGet, apiClient } from './http-client'

export interface AuthResponse {
  token: string
  user: { id: string; username: string; email: string }
}

const BASE = apiClient.defaults.baseURL || 'http://localhost:8000'

export const authController = {
  async login(username: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${BASE}/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || 'Login failed')
    }
    return res.json()
  },

  async register(username: string, email: string, password: string): Promise<AuthResponse> {
    const res = await fetch(`${BASE}/auth/register`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || 'Registration failed')
    }
    return res.json()
  },

  async logout(): Promise<void> {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
  },

  getToken(): string | null {
    return localStorage.getItem('auth_token')
  },

  setToken(token: string): void {
    localStorage.setItem('auth_token', token)
  },
}
