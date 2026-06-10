import { API_URL } from './config'

export class ApiError extends Error {
  status: number
  data: unknown

  constructor(status: number, message: string, data?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
  }
}

interface RequestOptions extends RequestInit {
  noAuth?: boolean
  silent?: boolean
  retries?: number
}

function getToken(): string | null {
  try {
    const { useAuthStore } = require('@/stores/auth-store')
    return useAuthStore.getState().token
  } catch {
    return null
  }
}

const RETRY_STATUSES = [408, 429, 502, 503, 504]
const MAX_RETRIES = 2
const BACKOFF_MS = [500, 1000]

async function fetchWithRetry(
  url: string,
  options: RequestOptions = {},
  attempt = 0
): Promise<Response> {
  const { noAuth, silent, retries = MAX_RETRIES, ...fetchOpts } = options

  if (!noAuth) {
    const token = getToken()
    if (token) {
      fetchOpts.headers = {
        ...fetchOpts.headers,
        Authorization: `Bearer ${token}`,
      }
    }
  }

  if (fetchOpts.body && typeof fetchOpts.body === 'string') {
    const headers = (fetchOpts.headers || {}) as Record<string, string>
    if (!headers['Content-Type']) {
      headers['Content-Type'] = 'application/json'
    }
    fetchOpts.headers = headers
  }

  try {
    const response = await fetch(url, fetchOpts)

    if (RETRY_STATUSES.includes(response.status) && attempt < retries) {
      await new Promise((r) => setTimeout(r, BACKOFF_MS[attempt] ?? 1000))
      return fetchWithRetry(url, options, attempt + 1)
    }

    return response
  } catch (error) {
    if (attempt < retries) {
      await new Promise((r) => setTimeout(r, BACKOFF_MS[attempt] ?? 1000))
      return fetchWithRetry(url, options, attempt + 1)
    }
    throw error
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let data: unknown
    try {
      data = await response.json()
    } catch {
      data = await response.text().catch(() => null)
    }
    const message =
      (data as Record<string, string>)?.detail ||
      (data as Record<string, string>)?.error ||
      `HTTP ${response.status}`
    throw new ApiError(response.status, message, data)
  }

  const contentType = response.headers.get('content-type')
  if (contentType?.includes('application/json')) {
    return response.json()
  }
  return response.text() as unknown as T
}

export async function apiGet<T>(path: string, opts?: RequestOptions): Promise<T> {
  const response = await fetchWithRetry(`${API_URL}${path}`, { ...opts, method: 'GET' })
  return handleResponse<T>(response)
}

export async function apiPost<T>(
  path: string,
  body?: unknown,
  opts?: RequestOptions
): Promise<T> {
  const response = await fetchWithRetry(`${API_URL}${path}`, {
    ...opts,
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  })
  return handleResponse<T>(response)
}

export async function apiPut<T>(
  path: string,
  body?: unknown,
  opts?: RequestOptions
): Promise<T> {
  const response = await fetchWithRetry(`${API_URL}${path}`, {
    ...opts,
    method: 'PUT',
    body: body ? JSON.stringify(body) : undefined,
  })
  return handleResponse<T>(response)
}

export async function apiPatch<T>(
  path: string,
  body?: unknown,
  opts?: RequestOptions
): Promise<T> {
  const response = await fetchWithRetry(`${API_URL}${path}`, {
    ...opts,
    method: 'PATCH',
    body: body ? JSON.stringify(body) : undefined,
  })
  return handleResponse<T>(response)
}

export async function apiDelete<T>(path: string, opts?: RequestOptions): Promise<T> {
  const response = await fetchWithRetry(`${API_URL}${path}`, { ...opts, method: 'DELETE' })
  return handleResponse<T>(response)
}
