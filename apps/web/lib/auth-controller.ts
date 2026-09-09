import { apiGet, apiPost } from './http-client'

export interface UserInfo {
  id: string
  username: string
  email: string
  role: string
  status: string
  tenant_id: string
}

export interface WorkspaceInfo {
  id: string
  name: string
  tenant_id: string
  description: string
  role: string
}

export interface AuthResponse {
  token: string
  user: UserInfo
}

export interface VerifyResponse {
  valid: boolean
  subject?: string
  expires?: number
}

export const authController = {
  async getMe(token: string): Promise<UserInfo> {
    return apiGet<UserInfo>('/auth/me', undefined, {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  async login(username: string, password: string): Promise<AuthResponse> {
    return apiPost<AuthResponse>('/auth/login', { username, password })
  },

  async register(username: string, email: string, password: string): Promise<AuthResponse> {
    return apiPost<AuthResponse>('/auth/register', { username, email, password })
  },

  async verify(token: string): Promise<VerifyResponse> {
    return apiPost<VerifyResponse>('/auth/verify', undefined, {
      headers: { Authorization: `Bearer ${token}` },
    })
  },

  async getWorkspaces(token: string): Promise<WorkspaceInfo[]> {
    try {
      const resp = await apiGet<{ data: WorkspaceInfo[] }>('/workspaces', undefined, {
        headers: { Authorization: `Bearer ${token}` },
      })
      return resp.data ?? []
    } catch {
      return []
    }
  },
}
