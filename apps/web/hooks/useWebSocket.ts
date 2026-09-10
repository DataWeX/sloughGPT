'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { trackEvent } from '@/lib/dev-log'

export type WebSocketStatus = 'disconnected' | 'connecting' | 'authenticating' | 'connected' | 'error'

export interface WebSocketMessage {
  type: string
  data: unknown
  raw?: Record<string, unknown>
}

export interface GenerateTokenEvent {
  token: string
}

export interface GenerateCompleteEvent {
  done: true
  status: 'done'
  text: string
}

export interface GenerateErrorEvent {
  status: 'error'
  error: string
}

export interface UseWebSocketOptions {
  url?: string
  autoReconnect?: boolean
  maxReconnectAttempts?: number
  reconnectBaseDelayMs?: number
  reconnectMaxDelayMs?: number
  onToken?: (token: string) => void
  onComplete?: (text: string) => void
  onError?: (error: string) => void
  onStatusChange?: (status: WebSocketStatus) => void
}

export interface UseWebSocketResult {
  status: WebSocketStatus
  error: string | null
  connect: (apiKey: string) => void
  disconnect: () => void
  sendGenerate: (prompt: string, options?: GenerateOptions) => void
  lastMessage: WebSocketMessage | null
}

export interface GenerateOptions {
  maxTokens?: number
  temperature?: number
  topP?: number
  topK?: number
  repetitionPenalty?: number
  model?: string
}

function getDefaultWsUrl(): string {
  if (typeof window === 'undefined') return 'ws://localhost:8000'
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketResult {
  const {
    url = getDefaultWsUrl(),
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectBaseDelayMs = 1000,
    reconnectMaxDelayMs = 30000,
    onToken,
    onComplete,
    onError,
    onStatusChange,
  } = options

  const [status, setStatus] = useState<WebSocketStatus>('disconnected')
  const [error, setError] = useState<string | null>(null)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const apiKeyRef = useRef<string>('')
  const mountedRef = useRef(true)
  const callbacksRef = useRef({ onToken, onComplete, onError, onStatusChange })

  callbacksRef.current = { onToken, onComplete, onError, onStatusChange }

  const updateStatus = useCallback((next: WebSocketStatus) => {
    if (!mountedRef.current) return
    setStatus(next)
    callbacksRef.current.onStatusChange?.(next)
  }, [])

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (wsRef.current !== null) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onerror = null
      wsRef.current.onclose = null
      try { wsRef.current.close() } catch { /* ignore */ }
      wsRef.current = null
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (!autoReconnect || !mountedRef.current) return
    const attempt = reconnectAttemptsRef.current
    if (attempt >= maxReconnectAttempts) {
      updateStatus('error')
      setError(`Reconnect failed after ${maxReconnectAttempts} attempts`)
      return
    }
    const delay = Math.min(reconnectBaseDelayMs * Math.pow(2, attempt), reconnectMaxDelayMs)
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null
      if (mountedRef.current && apiKeyRef.current) {
        reconnectAttemptsRef.current++
        connectInternal(apiKeyRef.current)
      }
    }, delay)
  }, [autoReconnect, maxReconnectAttempts, reconnectBaseDelayMs, reconnectMaxDelayMs, updateStatus])

  const handleAuthMessage = useCallback((data: Record<string, unknown>) => {
    if (data.status === 'authenticated') {
      updateStatus('connected')
      setError(null)
      reconnectAttemptsRef.current = 0
      trackEvent('ws_authenticated')
    } else {
      const errMsg = typeof data.error === 'string' ? data.error : 'Authentication failed'
      updateStatus('error')
      setError(errMsg)
      callbacksRef.current.onError?.(errMsg)
      trackEvent('ws_auth_error', { error: errMsg })
    }
  }, [updateStatus])

  const handleTokenMessage = useCallback((data: Record<string, unknown>) => {
    const token = typeof data.token === 'string' ? data.token : ''
    callbacksRef.current.onToken?.(token)
    setLastMessage({ type: 'token', data: token, raw: data })
  }, [])

  const handleDoneMessage = useCallback((data: Record<string, unknown>) => {
    const text = typeof data.text === 'string' ? data.text : ''
    callbacksRef.current.onComplete?.(text)
    setLastMessage({ type: 'complete', data: text, raw: data })
  }, [])

  const handleErrorMessage = useCallback((data: Record<string, unknown>) => {
    const errMsg = typeof data.error === 'string' ? data.error : 'Unknown error'
    callbacksRef.current.onError?.(errMsg)
    setLastMessage({ type: 'error', data: errMsg, raw: data })
  }, [])

  const onMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data)
      setLastMessage({ type: 'unknown', data, raw: data })

      if (data.status === 'authenticated' || (data.status === 'error' && wsRef.current?.readyState === WebSocket.OPEN)) {
        const ws = wsRef.current
        if (ws && ws.readyState === WebSocket.OPEN) {
          handleAuthMessage(data)
          return
        }
      }

      if ('token' in data && typeof data.token === 'string') {
        handleTokenMessage(data)
        return
      }

      if (data.done === true && data.status === 'done') {
        handleDoneMessage(data)
        return
      }

      if (data.status === 'error' && 'error' in data) {
        handleErrorMessage(data)
        return
      }

      if (data.type === 'pong') return
    } catch {
      setLastMessage({ type: 'raw', data: event.data })
    }
  }, [handleAuthMessage, handleTokenMessage, handleDoneMessage, handleErrorMessage])

  const connectInternal = useCallback((apiKey: string) => {
    cleanup()
    updateStatus('connecting')
    setError(null)
    apiKeyRef.current = apiKey

    const wsUrl = `${url}/ws/generate`
    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl)
    } catch (e) {
      updateStatus('error')
      setError(e instanceof Error ? e.message : 'Failed to create WebSocket')
      scheduleReconnect()
      return
    }
    wsRef.current = ws

    ws.onopen = () => {
      updateStatus('authenticating')
      try {
        ws.send(JSON.stringify({ api_key: apiKey }))
      } catch (e) {
        updateStatus('error')
        setError(e instanceof Error ? e.message : 'Failed to send auth')
      }
    }

    ws.onmessage = onMessage

    ws.onerror = () => {
      updateStatus('error')
      setError('WebSocket connection error')
      trackEvent('ws_error')
    }

    ws.onclose = () => {
      if (wsRef.current === ws) {
        wsRef.current = null
        if (mountedRef.current && apiKeyRef.current) {
          updateStatus('connecting')
          scheduleReconnect()
        } else {
          updateStatus('disconnected')
        }
      }
    }
  }, [url, cleanup, updateStatus, scheduleReconnect, onMessage])

  const connect = useCallback((apiKey: string) => {
    reconnectAttemptsRef.current = 0
    connectInternal(apiKey)
  }, [connectInternal])

  const disconnect = useCallback(() => {
    apiKeyRef.current = ''
    reconnectAttemptsRef.current = 0
    cleanup()
    updateStatus('disconnected')
    setError(null)
  }, [cleanup, updateStatus])

  const sendGenerate = useCallback((prompt: string, genOptions: GenerateOptions = {}) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('Not connected')
      return
    }
    const payload = {
      prompt,
      max_tokens: genOptions.maxTokens ?? 256,
      temperature: genOptions.temperature ?? 0.7,
      top_p: genOptions.topP ?? 0.85,
      top_k: genOptions.topK ?? 40,
      repetition_penalty: genOptions.repetitionPenalty ?? 1.15,
      ...(genOptions.model ? { model: genOptions.model } : {}),
    }
    try {
      ws.send(JSON.stringify(payload))
      trackEvent('ws_generate_sent', { prompt_length: prompt.length })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to send')
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      cleanup()
    }
  }, [cleanup])

  return { status, error, connect, disconnect, sendGenerate, lastMessage }
}
