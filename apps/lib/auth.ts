import { _getToken } from './ports'
import { _corrId, _trackCorrId } from './corr-id'
import { _resolveUrl } from './url'

interface AuthFetchOptions extends RequestInit {
  noAuth?: boolean
}

export async function authFetch(url: string, opts?: AuthFetchOptions): Promise<Response> {
  const corrId = _corrId()
  const headers: Record<string, string> = {}
  if (!opts?.noAuth) {
    const token = _getToken()
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
  // Base-URL policy lives here with every other transport path (client, request,
  // fetchSSE): relative = "the API host" (or the gateway relay in a same-origin
  // build, where PUBLIC_API_URL is ''), absolute passes through untouched, and
  // /api/planner|calendar stay same-origin.
  const fullUrl = _resolveUrl(url)
  _trackCorrId(corrId, fullUrl)
  return fetch(fullUrl, { ...opts, headers })
}
