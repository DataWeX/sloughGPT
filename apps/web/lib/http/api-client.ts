import { PUBLIC_API_URL } from '../config'
import { apiGet, apiPost, apiPut, apiDelete, apiPatch } from './verbs'

interface ApiClientConfig {
  signal?: AbortSignal
  params?: Record<string, string>
  _silent?: boolean
  _noAuth?: boolean
  _raw?: boolean
}

function createApiClient(baseURL?: string) {
  const prefix = baseURL ?? PUBLIC_API_URL
  return {
    defaults: { baseURL: prefix },
    get: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiGet<T>(u, config?.params, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    post: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPost<T>(u, body, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
        raw: config?._raw,
      })
    },
    put: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPut<T>(u, body, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    delete: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiDelete<T>(u, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    patch: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${prefix}${url}` : url
      return apiPatch<T>(u, body, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
  }
}

export const apiClient = createApiClient()
export { createApiClient }
