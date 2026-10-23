import { _getBaseUrl } from './ports'
import { apiGet, apiPost, apiPut, apiDelete, apiPatch } from './verbs'

interface ApiClientConfig {
  signal?: AbortSignal
  params?: Record<string, string>
  _silent?: boolean
  _noAuth?: boolean
  _raw?: boolean
}

function createApiClient(baseURL?: string) {
  // Read the port at CALL time — apiClient is built at module load, before
  // the host wires the transport, so a captured prefix would freeze the
  // pre-wiring default for the whole app lifetime.
  const resolveBase = (): string => baseURL ?? _getBaseUrl()
  return {
    defaults: {
      get baseURL(): string {
        return resolveBase()
      },
    },
    get: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${resolveBase()}${url}` : url
      return apiGet<T>(u, config?.params, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    post: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${resolveBase()}${url}` : url
      return apiPost<T>(u, body, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
        raw: config?._raw,
      })
    },
    put: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${resolveBase()}${url}` : url
      return apiPut<T>(u, body, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    delete: <T>(url: string, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${resolveBase()}${url}` : url
      return apiDelete<T>(u, {
        signal: config?.signal,
        silent: config?._silent,
        noAuth: config?._noAuth,
      })
    },
    patch: <T>(url: string, body?: unknown, config?: ApiClientConfig): Promise<T> => {
      const u = url.startsWith('/') ? `${resolveBase()}${url}` : url
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
