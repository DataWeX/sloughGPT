import { logger } from '../dev-log'
import { useAuthStore } from '../auth'
import { consumeSSEChunk, consumeSSEDrain } from '../sse-client'
import { _corrId, _trackCorrId } from './corr-id'
import { _resolveUrl } from './url'

export interface SSEEvent {
  stream?: string
  phase?: string
  status?: string
  data?: Record<string, unknown>
  meta?: Record<string, unknown>
  message?: string
  error?: string
  id?: string
}

interface StreamSSEOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  noAuth?: boolean
  lastEventId?: string
  maxRetries?: number
}

export async function* streamSSE(url: string, opts?: StreamSSEOptions): AsyncGenerator<SSEEvent> {
  const method = opts?.method ?? 'POST'
  const maxRetries = opts?.maxRetries ?? 3
  const baseDelay = 500

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const corrId = _corrId()
    const headers: Record<string, string> = {}
    if (method !== 'GET') headers['Content-Type'] = 'application/json'
    if (!opts?.noAuth) {
      const token = useAuthStore.getState().token
      if (token) headers['Authorization'] = `Bearer ${token}`
    }
    if (opts?.lastEventId) {
      headers['Last-Event-ID'] = opts.lastEventId
    }
    headers['X-Correlation-ID'] = corrId

    _trackCorrId(corrId, url)

    logger.debug(`>>> SSE ${method} ${url} corr=${corrId}`, { corrId, method, url })

    let res: Response
    try {
      res = await fetch(_resolveUrl(url), {
        method,
        headers,
        body: opts?.body != null ? JSON.stringify(opts.body) : undefined,
        signal: opts?.signal,
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Network error'
      const aborted =
        opts?.signal?.aborted === true || (e instanceof Error && e.name === 'AbortError')
      if (aborted) {
        logger.debug(`<<< SSE ${method} ${url} aborted corr=${corrId}`, { corrId, reason: msg })
      } else {
        logger.error(`<<< SSE ${method} ${url} FAILED corr=${corrId}: ${msg}`, { corrId })
      }
      yield { status: 'error', message: `Connection error: ${msg}` }
      return
    }

    logger.debug(`<<< SSE ${method} ${url} ${res.status} corr=${corrId}`, {
      corrId,
      status: res.status,
    })

    if (!res.ok) {
      const status = res.status
      if (status === 503 && attempt < maxRetries) {
        const delay = baseDelay * Math.pow(2, attempt)
        logger.warning(
          `SSE retry ${attempt + 1}/${maxRetries} ${method} ${url} ${status} delay=${delay}ms corr=${corrId}`,
          { corrId, method, url, status, attempt, maxRetries, delay },
        )
        await new Promise((r) => setTimeout(r, delay))
        continue
      }
      yield {
        status: 'error',
        message: `HTTP ${res.status}${res.statusText ? `: ${res.statusText}` : ''}`,
        data: { http_status: res.status, error: `HTTP ${res.status}` },
      }
      return
    }

    if (!res.body) {
      yield { status: 'error', message: 'No response body' }
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    try {
      while (true) {
        let chunk: ReadableStreamReadResult<Uint8Array>
        try {
          chunk = await reader.read()
        } catch (e) {
          if (opts?.signal?.aborted === true || (e instanceof Error && e.name === 'AbortError')) {
            yield { status: 'error', message: 'Stream aborted' }
            return
          }
          const msg = e instanceof Error ? e.message : 'Read error'
          yield { status: 'error', message: `Stream disconnected: ${msg}` }
          return
        }
        if (chunk.done) break
        const { events, buffer: nextBuffer } = consumeSSEChunk(
          buffer,
          decoder,
          chunk.value,
          (payload) =>
            logger.warning('SSE malformed JSON payload skipped', {
              payload,
              exception: 'parse error',
            }),
        )
        buffer = nextBuffer
        for (const event of events) {
          yield event as SSEEvent
        }
      }
      // Drain remaining partial buffer
      const drained = consumeSSEDrain(buffer)
      for (const event of drained) {
        yield event as SSEEvent
      }
      return
    } finally {
      reader.releaseLock()
    }
  }
}
