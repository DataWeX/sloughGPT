import { API_URL } from './config'

export interface SSEEnvelope {
  stream: string
  phase: string
  status: 'working' | 'success' | 'error' | 'complete'
  data: Record<string, unknown>
  meta?: Record<string, unknown>
  message?: string
}

export interface SSETokenEvent {
  token: string
  done: boolean
  error?: string
  meta?: Record<string, unknown>
}

function getToken(): string | null {
  try {
    const { useAuthStore } = require('@/stores/auth-store')
    return useAuthStore.getState().token
  } catch {
    return null
  }
}

export async function* streamSSE(
  path: string,
  body: Record<string, unknown>,
  signal?: AbortSignal
): AsyncGenerator<SSETokenEvent> {
  const token = getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => `HTTP ${response.status}`)
    yield { token: '', done: true, error: errorText }
    return
  }

  if (!response.body) {
    yield { token: '', done: true, error: 'No response body' }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || !trimmed.startsWith('data: ')) continue

        const jsonStr = trimmed.slice(6)
        if (!jsonStr || jsonStr === '[DONE]') continue

        try {
          const envelope: SSEEnvelope = JSON.parse(jsonStr)

          if (envelope.status === 'error') {
            const errorMsg =
              (envelope.data?.error as string) || envelope.message || 'Stream error'
            yield { token: '', done: true, error: errorMsg, meta: envelope.meta }
            return
          }

          if (envelope.status === 'complete') {
            yield { token: '', done: true, meta: envelope.meta }
            return
          }

          const token = envelope.data?.token as string
          if (token !== undefined) {
            yield { token, done: false, meta: envelope.meta }
          }
        } catch {
          // skip malformed JSON lines
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  yield { token: '', done: true }
}

export function createAbortController(): AbortController {
  return new AbortController()
}
