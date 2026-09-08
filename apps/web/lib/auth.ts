'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { trackEvent } from '@/lib/dev-log'

interface User {
  id: string
  username: string
  email: string
  role: string
  status: string
  tenant_id: string
}

interface Workspace {
  id: string
  name: string
  tenant_id: string
  description: string
  role: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  workspaces: Workspace[]
  currentWorkspace: Workspace | null
  login: (user: User, token: string) => void
  logout: () => void
  setUser: (user: User | null) => void
  setWorkspaces: (workspaces: Workspace[]) => void
  switchWorkspace: (workspaceId: string) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      workspaces: [],
      currentWorkspace: null,
      login: (user, token) => {
        set({ user, token, isAuthenticated: true })
        trackEvent('auth_login', { user_id: user?.id ?? user?.username ?? 'unknown' })
      },
      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          workspaces: [],
          currentWorkspace: null,
        })
        trackEvent('auth_logout')
      },
      setUser: (user) => {
        set({ user })
        trackEvent('auth_user_changed', { user_id: user?.id ?? user?.username ?? 'unknown' })
      },
      setWorkspaces: (workspaces) => {
        set({ workspaces })
        // Auto-select first workspace if none selected
        const current = get().currentWorkspace
        if (!current && workspaces.length > 0) {
          set({ currentWorkspace: workspaces[0] })
        } else if (current) {
          // Update current workspace data if it still exists
          const updated = workspaces.find((w) => w.id === current.id)
          if (updated) {
            set({ currentWorkspace: updated })
          } else if (workspaces.length > 0) {
            set({ currentWorkspace: workspaces[0] })
          } else {
            set({ currentWorkspace: null })
          }
        }
      },
      switchWorkspace: (workspaceId) => {
        const workspace = get().workspaces.find((w) => w.id === workspaceId)
        if (workspace) {
          set({ currentWorkspace: workspace })
          trackEvent('workspace_switch', { workspace_id: workspaceId, workspace_name: workspace.name })
        }
      },
    }),
    {
      name: 'man-auth',
    }
  )
)
