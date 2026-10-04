import { useAuthStore } from '../auth'
import { _corrId, _trackCorrId } from './corr-id'

interface AuthFetchOptions extends RequestInit {
  noAuth?: boolean
}

export async function authFetch(url: string, opts?: AuthFetchOptions): Promise<Response> {
  const corrId = _corrId()
  const headers: Record<string, string> = {}
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  if (opts?.headers) {
    if (opts.headers instanceof Headers) {
      opts.headers.forEach((v, k) => {
        headers[k] = v
      })
    } else {
      Object.assign(headers, opts.headers)
    }
  }
  headers['X-Correlation-ID'] = corrId
  _trackCorrId(corrId, url)
  return fetch(url, { ...opts, headers })
}
