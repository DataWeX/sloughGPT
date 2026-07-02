export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown,
    public requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

export interface RequestOptions {
  signal?: AbortSignal
  timeout?: number
  noAuth?: boolean
  raw?: boolean
  onProgress?: (pct: number) => void
  headers?: Record<string, string>
  silent?: boolean
}

const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY = 500
const DEFAULT_TIMEOUT_MS = 15_000

async function request<T>(
  method: string,
  url: string,
  body?: unknown,
  opts?: RequestOptions,
): Promise<T> {
  const apiUrl = `${PUBLIC_API_URL}${url}`
  const headers: Record<string, string> = {}
  if (!opts?.raw) headers['Content-Type'] = 'application/json'
  if (opts?.headers) Object.assign(headers, opts.headers)
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const timeoutMs = opts?.timeout ?? DEFAULT_TIMEOUT_MS
  let retries = 0
  while (true) {
    let signal = opts?.signal
    let timer: ReturnType<typeof setTimeout> | undefined
    if (!signal) {
      const ac = new AbortController()
      timer = setTimeout(() => ac.abort(), timeoutMs)
      signal = ac.signal
    }
    try {
      const res = await fetch(apiUrl, {
        method,
        headers,
        body: opts?.raw ? (body as BodyInit) : body != null ? JSON.stringify(body) : undefined,
        signal,
      })
      if (timer) clearTimeout(timer)

      if (!res.ok) {
        const status = res.status
        const isRetryable = RETRYABLE_STATUSES.has(status)
        if (isRetryable && retries < MAX_RETRIES) {
          retries++
          const retryAfter = Number(res.headers.get('Retry-After')) || 0
          const delay = retryAfter > 0 ? retryAfter * 1000 : BASE_DELAY * Math.pow(2, retries - 1)
          await new Promise(r => setTimeout(r, delay))
          continue
        }

        const text = await res.text()
        const requestId = res.headers.get('X-Request-ID') || undefined
        let detail: string | undefined
        try { const j = JSON.parse(text); detail = j.detail ?? j.message ?? j.error }
        catch { detail = text || res.statusText }
        const message = Array.isArray(detail) ? detail.map((d: any) => d.msg).join('; ') : detail || 'Request failed'

        if (!opts?.silent) {
          const apiErr = new ApiError(message, status, { raw: text }, requestId)
          import('./error-store').then(({ useErrorStore }) => {
            useErrorStore.getState().addError(apiErr, {
              source: url,
              title: status >= 500 ? 'Server Error' : status >= 400 ? `HTTP ${status}` : 'API Error',
              requestId: requestId,
            })
          })
        }
        throw new ApiError(message, status, { raw: text }, requestId)
      }

      const text = await res.text()
      if (!text) return undefined as T
      return JSON.parse(text) as T
    } catch (e: any) {
      if (timer) clearTimeout(timer)
      if (e instanceof ApiError) throw e
      const status = 0
      const isTimeout = e.name === 'AbortError' || e.message?.includes('aborted')
      const message = isTimeout
        ? `Request timed out after ${timeoutMs / 1000}s`
        : e.message === 'Failed to fetch' ? 'Connection unavailable' : (e.message || 'Request failed')

      if (!opts?.silent) {
        const apiErr = new ApiError(message, status)
        import('./error-store').then(({ useErrorStore }) => {
          useErrorStore.getState().addError(apiErr, {
            source: url,
            title: 'Connection Error',
          })
        })
      }

      if (retries < MAX_RETRIES) {
        retries++
        await new Promise(r => setTimeout(r, BASE_DELAY * Math.pow(2, retries - 1)))
        continue
      }
      throw new ApiError(message, status)
    }
  }
}

export async function apiGet<T>(url: string, params?: Record<string, string>, opts?: RequestOptions): Promise<T> {
  const qs = params ? '?' + new URLSearchParams(params).toString() : ''
  return request<T>('GET', url + qs, undefined, opts)
}

export async function apiPost<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('POST', url, body, opts)
}

export async function apiPut<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('PUT', url, body, opts)
}

export async function apiDelete<T>(url: string, opts?: RequestOptions): Promise<T> {
  return request<T>('DELETE', url, undefined, opts)
}

export async function apiPatch<T>(url: string, body?: unknown, opts?: RequestOptions): Promise<T> {
  return request<T>('PATCH', url, body, opts)
}

function createApiClient(baseURL?: string) {
  const prefix = baseURL ?? PUBLIC_API_URL
  return {
    defaults: { baseURL: prefix },
    get: <T>(url: string, config?: any) => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiGet<T>(u, config?.params, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth }) as any
    },
    post: <T>(url: string, body?: any, config?: any) => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPost<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth, raw: config?._raw }) as any
    },
    put: <T>(url: string, body?: any, config?: any) => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPut<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth }) as any
    },
    delete: <T>(url: string, config?: any) => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiDelete<T>(u, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth }) as any
    },
    patch: <T>(url: string, body?: any, config?: any) => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPatch<T>(u, body, { signal: config?.signal, silent: config?._silent, noAuth: config?._noAuth }) as any
    },
  }
}

export const apiClient = createApiClient()
export { createApiClient }
