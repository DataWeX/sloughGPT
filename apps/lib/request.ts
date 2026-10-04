import { logger } from '../web/lib/dev-log'
import { useAuthStore } from '../web/lib/auth'
import { ApiError } from './errors'
import type { RequestConfig, ResponseEnvelope, RequestOptions } from './types'
import { _corrId, _trackCorrId } from './corr-id'
import { _resolveUrl, _isDocstoreUrl } from './url'
import {
  RETRYABLE_STATUSES,
  DOCSTORE_RETRYABLE_STATUSES,
  DOCSTORE_MAX_RETRIES,
  DOCSTORE_BASE_DELAY,
  httpDefaults,
} from './retry'
import { _globalInterceptors, _globalCircuitBreaker, _globalHooks } from './state'

export async function request<T>(
  method: string,
  url: string,
  body?: unknown,
  opts?: RequestOptions,
  // Internal: pre-built config from httpClient
  _preBuiltConfig?: RequestConfig,
): Promise<T> {
  const startTime = Date.now()
  const corrId = _preBuiltConfig?.correlationId ?? _corrId()
  _trackCorrId(corrId, url)

  const headers: Record<string, string> = {}
  if (!opts?.raw) headers['Content-Type'] = 'application/json'
  if (opts?.headers) Object.assign(headers, opts.headers)
  if (!opts?.noAuth) {
    const token = useAuthStore.getState().token
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  headers['X-Correlation-ID'] = corrId

  const fullUrl = _resolveUrl(url)

  const config: RequestConfig = _preBuiltConfig ?? {
    method,
    url,
    fullUrl,
    headers,
    body,
    signal: opts?.signal,
    correlationId: corrId,
    startTime,
    opts: opts ?? {},
  }

  // Run request interceptors
  const finalConfig = await _globalInterceptors.request.run(config)

  logger.debug(`>>> ${method} ${url} corr=${corrId}`, { corrId, method, url })

  const timeoutMs = opts?.timeout ?? httpDefaults.timeout
  const isDocstore = _isDocstoreUrl(url)
  const maxRetries = isDocstore
    ? DOCSTORE_MAX_RETRIES
    : opts?.skipCircuitBreaker
      ? 0
      : httpDefaults.maxRetries
  const baseDelay = isDocstore ? DOCSTORE_BASE_DELAY : httpDefaults.baseDelay
  const retryableStatuses = isDocstore ? DOCSTORE_RETRYABLE_STATUSES : RETRYABLE_STATUSES
  let retries = 0

  for (;;) {
    let signal = opts?.signal
    let timer: ReturnType<typeof setTimeout> | undefined
    if (!signal) {
      const ac = new AbortController()
      timer = setTimeout(() => ac.abort(), timeoutMs)
      signal = ac.signal
    }
    try {
      // Upload progress: wrap body ReadableStream if needed
      let fetchBody: BodyInit | undefined
      if (opts?.raw) {
        fetchBody = body as BodyInit
      } else if (body != null) {
        const jsonStr = JSON.stringify(body)
        if (opts?.progress?.onUpload) {
          const blob = new Blob([jsonStr], { type: 'application/json' })
          fetchBody = new ReadableStream({
            start(controller) {
              const chunk = new Uint8Array(blob.size)
              blob.arrayBuffer().then((ab) => {
                chunk.set(new Uint8Array(ab))
                controller.enqueue(chunk)
                controller.close()
                opts!.progress!.onUpload!(100, blob.size, blob.size)
              })
            },
          })
        } else {
          fetchBody = jsonStr
        }
      }

      let res: Response
      if (opts?.progress?.onDownload) {
        const originalRes = await fetch(finalConfig.fullUrl, {
          method,
          headers: finalConfig.headers,
          body: fetchBody,
          signal,
        })
        if (!originalRes.ok || !originalRes.body) {
          res = originalRes
        } else {
          const contentLength = Number(originalRes.headers.get('content-length')) || 0
          let loaded = 0
          const reader = originalRes.body.getReader()
          const chunks: Uint8Array[] = []
          const decoder = new TextDecoder()
          for (;;) {
            const { done, value } = await reader.read()
            if (done) break
            chunks.push(value)
            loaded += value.length
            if (contentLength > 0) {
              opts.progress.onDownload(
                Math.round((loaded / contentLength) * 100),
                loaded,
                contentLength,
              )
            }
          }
          reader.releaseLock()
          const allChunks = new Uint8Array(loaded)
          let offset = 0
          for (const chunk of chunks) {
            allChunks.set(chunk, offset)
            offset += chunk.length
          }
          const text = decoder.decode(allChunks)
          res = new Response(text, {
            status: originalRes.status,
            statusText: originalRes.statusText,
            headers: originalRes.headers,
          })
        }
      } else {
        res = await fetch(finalConfig.fullUrl, {
          method,
          headers: finalConfig.headers,
          body: fetchBody,
          signal,
        })
      }

      if (timer) clearTimeout(timer)

      logger.debug(`<<< ${method} ${url} ${res.status} corr=${corrId}`, {
        corrId,
        method,
        url,
        status: res.status,
      })

      // Build response envelope
      const resText = await res.text()
      const envelope: ResponseEnvelope = {
        ok: res.ok,
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
        text: resText,
        config: finalConfig,
      }
      if (res.ok && resText) {
        try {
          envelope.json = JSON.parse(resText)
        } catch {
          /* not JSON */
        }
      }

      // Run response interceptors
      const finalEnvelope = await _globalInterceptors.response.run(envelope)

      if (!finalEnvelope.ok) {
        const status = finalEnvelope.status
        const isRetryable = retryableStatuses.has(status)
        if (isRetryable && retries < maxRetries) {
          retries++
          const retryAfter = Number(finalEnvelope.headers.get('Retry-After')) || 0
          const delay = retryAfter > 0 ? retryAfter * 1000 : baseDelay * Math.pow(2, retries - 1)
          logger.warning(
            `retry ${retries}/${maxRetries} ${method} ${url} ${status} delay=${delay}ms corr=${corrId}`,
            { corrId, method, url, status, retries, maxRetries, delay },
          )
          await new Promise((r) => setTimeout(r, delay))
          continue
        }

        let detail: string | undefined
        let errorCode: string | undefined
        let errorDetails: unknown | undefined
        let correlationId: string | undefined
        try {
          const j = JSON.parse(finalEnvelope.text)
          detail = j.detail ?? j.message ?? j.error ?? finalEnvelope.text
          errorCode = j.code
          errorDetails = j.details
          correlationId = j.correlation_id
        } catch {
          detail = finalEnvelope.text || finalEnvelope.statusText || 'Could not request'
        }
        const message = Array.isArray(detail)
          ? detail
              .map((d: { msg?: string } | string) => (typeof d === 'string' ? d : (d.msg ?? '')))
              .join('; ')
          : detail || 'Could not request'

        const requestId = finalEnvelope.headers.get('X-Request-ID') || undefined
        const apiErr = new ApiError(
          message,
          status,
          { raw: finalEnvelope.text, code: errorCode, details: errorDetails, correlationId },
          requestId,
        )

        // Circuit breaker
        if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordFailure()

        // Lifecycle: onError
        for (const hook of _globalHooks.onError) {
          try {
            await hook(apiErr, finalConfig)
          } catch {
            /* hook errors swallowed */
          }
        }

        if (!opts?.silent) {
          import('../web/lib/error-store').then(({ useErrorStore }) => {
            useErrorStore.getState().addError(apiErr, {
              source: url,
              title:
                status >= 500 ? 'Server Error' : status >= 400 ? `HTTP ${status}` : 'API Error',
              requestId,
            })
          })
        }
        throw apiErr
      }

      // Success — circuit breaker
      if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordSuccess()

      // Return typed result
      if (!finalEnvelope.text) return undefined as T
      const json = finalEnvelope.json ?? JSON.parse(finalEnvelope.text)
      if (json && typeof json === 'object' && 'status' in json && 'data' in json) {
        const result = json.data as T
        if (json.meta && typeof result === 'object' && result !== null) {
          Object.defineProperty(result, '_meta', { value: json.meta, enumerable: false })
        }
        return result
      }
      return json as T
    } catch (e: unknown) {
      if (timer) clearTimeout(timer)
      if (e instanceof ApiError) throw e
      if (opts?.signal?.aborted) throw e
      const status = 0
      const name = e instanceof Error ? e.name : undefined
      const message_ = e instanceof Error ? e.message : undefined
      const cause =
        e instanceof Error ? (e as Error & { cause?: { code?: string } }).cause : undefined
      const isTimeout = name === 'AbortError' || message_?.includes('aborted')
      const isConnRefused = message_ === 'Failed to fetch' || cause?.code === 'ECONNREFUSED'
      const message = isTimeout
        ? `Request timed out after ${timeoutMs / 1000}s`
        : isConnRefused
          ? 'Connection unavailable — server may be starting up'
          : message_ || 'Could not request'

      const kind = isTimeout ? 'timeout' : isConnRefused ? 'connection_refused' : 'unknown'

      if (retries < maxRetries) {
        retries++
        const delay = baseDelay * Math.pow(2, retries - 1)
        logger.warning(
          `retry ${retries}/${maxRetries} ${method} ${url} ${kind} delay=${delay}ms corr=${corrId}`,
          { corrId, method, url, kind, retries, maxRetries, delay },
        )
        await new Promise((r) => setTimeout(r, delay))
        continue
      }

      if (!opts?.skipCircuitBreaker) _globalCircuitBreaker.recordFailure()

      const apiErr = new ApiError(message, status)

      for (const hook of _globalHooks.onError) {
        try {
          await hook(apiErr, finalConfig)
        } catch {
          /* hook errors swallowed */
        }
      }

      if (!opts?.silent) {
        import('../web/lib/error-store').then(({ useErrorStore }) => {
          useErrorStore.getState().addError(apiErr, {
            source: url,
            title: 'Connection Error',
          })
        })
        import('../web/lib/api-monitor-store').then(({ useApiMonitor }) => {
          useApiMonitor.getState().addFailure({
            endpoint: url,
            error: message,
            status: 0,
            timeoutMs,
            timestamp: Date.now(),
            kind,
          })
        })
      }

      throw apiErr
    }
  }
}
