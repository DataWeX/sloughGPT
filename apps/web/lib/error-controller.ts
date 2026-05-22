/**
 * Error monitoring controller — log client errors and fetch recent error history.
 */

import { apiGet, apiPost, apiDelete } from './http-client'

export interface ErrorEntry {
  id: string
  message: string
  source: string
  stack: string | null
  url: string | null
  line: number | null
  col: number | null
  client_host: string
  timestamp: string
  metadata: Record<string, unknown>
}

export interface ErrorLogResponse {
  status: string
  logged: number
}

export interface RecentErrorsResponse {
  errors: ErrorEntry[]
  unread_count: number
}

export const errorController = {
  /** Fetch recent client-side errors from the backend ring buffer. */
  async getRecent(limit: number = 50): Promise<{ errors: ErrorEntry[]; unread_count: number }> {
    return apiGet<RecentErrorsResponse>('/errors/recent', { limit: String(limit) })
  },

  /** Report a single error to the backend. */
  async report(
    message: string,
    source: string = 'web',
    extra?: Partial<Omit<ErrorEntry, 'id' | 'client_host'>>,
  ): Promise<ErrorLogResponse> {
    return apiPost<ErrorLogResponse>('/errors/log', {
      errors: [
        {
          message,
          source,
          timestamp: new Date().toISOString(),
          url: typeof window !== 'undefined' ? window.location.href : undefined,
          ...extra,
        },
      ],
    })
  },

  /** Clear all stored errors and reset unread counter. */
  async clear(): Promise<void> {
    await apiDelete('/errors/clear')
  },

  /** Get unread error count. */
  async getUnreadCount(): Promise<number> {
    const res = await apiGet<{ unread_count: number }>('/errors/unread')
    return res.unread_count
  },
}
