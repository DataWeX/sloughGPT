'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { trackEvent } from '@/lib/dev-log'

interface User {
  id: string
  username: string
  email: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (user: User, token: string) => void
  logout: () => void
  setUser: (user: User | null) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => {
        set({ user, token, isAuthenticated: true })
        trackEvent('auth_login', { user_id: user?.id ?? user?.username ?? 'unknown' })
      },
      logout: () => {
        set({ user: null, token: null, isAuthenticated: false })
        trackEvent('auth_logout')
      },
      setUser: (user) => {
        set({ user })
        trackEvent('auth_user_changed', { user_id: user?.id ?? user?.username ?? 'unknown' })
      },
    }),
    {
      name: 'man-auth',
    }
  )
)
