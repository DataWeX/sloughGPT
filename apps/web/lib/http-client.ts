/**
 * Composable HTTP client — axios-based with interceptors.
 *
 * Usage:
 *   import { apiClient } from '@/lib/http-client'
 *
 *   const models = await apiClient.get<Model[]>('/models')
 *   const result = await apiClient.post<Model>('/models/load', { model_id: 'gpt2' })
 *
 * Auth token is automatically injected from useAuthStore.
 * Errors are normalized to ApiError with status code + data.
 * Retries on 5xx / 429 / network errors (3 attempts with backoff).
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios'
import { PUBLIC_API_URL } from './config'
import { useAuthStore } from './auth'

// ─── Error type ───────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ─── Request config extensions ─────────────────────────────────────────────────

export interface RequestOptions {
  /** AbortSignal for cancellation. */
  signal?: AbortSignal
  /** Custom timeout in ms (default: 30s). */
  timeout?: number
  /** Skip auth header injection. */
  noAuth?: boolean
  /** Raw FormData / Blob — skip JSON serialization. */
  raw?: boolean
  /** Progress callback for uploads. */
  onProgress?: (pct: number) => void
  /** Extra headers to merge into the request. */
  headers?: Record<string, string>
}

// ─── Client factory ────────────────────────────────────────────────────────────

const RETRYABLE_STATUSES = new Set([408, 429, 502, 503, 504])
const MAX_RETRIES = 2
const BASE_DELAY = 500

export function createApiClient(baseURL: string): AxiosInstance {
  const client: AxiosInstance = axios.create({
    baseURL,
    timeout: 30_000,
    headers: { 'Content-Type': 'application/json' },
  })

  // ── Request interceptor: inject auth token ─────────────────────────
  client.interceptors.request.use(
    (config) => {
      if ((config as any)._noAuth === true) return config
      const token = useAuthStore.getState().token
      if (token) {
        config.headers.set('Authorization', `Bearer ${token}`)
      }
      return config
    },
    (error) => Promise.reject(error),
  )

  // ── Response interceptor: normalise errors + retry ─────────────────
  let retryCount = 0
  client.interceptors.response.use(
    (response) => {
      retryCount = 0
      return response
    },
    async (error: AxiosError) => {
      const status = error.response?.status ?? 0
      const data = error.response?.data
      const isRetryable = status === 0 || RETRYABLE_STATUSES.has(status)

      if (isRetryable && retryCount < MAX_RETRIES) {
        retryCount++
        const delay = BASE_DELAY * Math.pow(2, retryCount - 1)
        await new Promise((r) => setTimeout(r, delay))
        return client.request(error.config!)
      }

      retryCount = 0
      const detail = (error.response?.data as any)?.detail
      const message = Array.isArray(detail)
        ? detail.map((d: any) => d.msg).join('; ')
        : detail ??
          error.response?.statusText ??
          (status === 0 ? 'Connection unavailable' : error.message) ??
          'Request failed'

      return Promise.reject(new ApiError(message, status, data))
    },
  )

  return client
}

// ── Shared singleton ───────────────────────────────────────────────────────────

export const apiClient = createApiClient(PUBLIC_API_URL)

// ── Typed request helpers (composable) ─────────────────────────────────────────

function buildConfig(opts?: RequestOptions): AxiosRequestConfig & { _noAuth?: boolean } {
  return {
    signal: opts?.signal,
    timeout: opts?.timeout,
    _noAuth: opts?.noAuth,
    ...(opts?.headers ? { headers: opts.headers } : {}),
  }
}

export async function apiGet<T>(
  url: string,
  params?: Record<string, string>,
  opts?: RequestOptions,
): Promise<T> {
  const res = await apiClient.get<T>(url, { ...buildConfig(opts), params })
  return res.data
}

export async function apiPost<T>(
  url: string,
  body?: unknown,
  opts?: RequestOptions,
): Promise<T> {
  if (opts?.raw) {
    const res = await apiClient.post<T>(url, body as any, {
      ...buildConfig(opts),
      headers: { ...opts.headers, 'Content-Type': 'multipart/form-data' },
      onUploadProgress: opts?.onProgress
        ? (e) => e.total && opts.onProgress!(Math.round((e.loaded / e.total) * 100))
        : undefined,
    })
    return res.data
  }
  const res = await apiClient.post<T>(url, body ?? null, buildConfig(opts))
  return res.data
}

export async function apiPut<T>(
  url: string,
  body?: unknown,
  opts?: RequestOptions,
): Promise<T> {
  const res = await apiClient.put<T>(url, body ?? null, buildConfig(opts))
  return res.data
}

export async function apiDelete<T>(
  url: string,
  opts?: RequestOptions,
): Promise<T> {
  const res = await apiClient.delete<T>(url, buildConfig(opts))
  return res.data
}

export async function apiPatch<T>(
  url: string,
  body?: unknown,
  opts?: RequestOptions,
): Promise<T> {
  const res = await apiClient.patch<T>(url, body ?? null, buildConfig(opts))
  return res.data
}


