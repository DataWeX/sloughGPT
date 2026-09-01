'use client'

/**
 * StreamingPipeline — manages the SSE → token buffer → render flow.
 *
 * This is the single integration point between the backend streaming API
 * and the frontend renderer. It owns:
 *   1. SSE connection lifecycle (connect, reconnect, abort)
 *   2. Token buffering and flush scheduling (RAF-based)
 *   3. Message state mutations (append, update, finalize)
 *   4. Tool call event aggregation
 *   5. Thinking/reasoning capture
 */

import { useCallback, useRef, useEffect, useState } from 'react'
import { streamChatResponse, type ToolCallEvent } from '@/lib/stream-chat-response'
import { logger } from '@/lib/dev-log'
import type { ChatMessage } from '@/lib/chat-utils'

const _log = logger.child('streaming-pipeline')

interface StreamingPipelineOptions {
  messages: ChatMessage[]
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
  model: string
  temperature: number
  maxTokens: number
  systemPrompt: string
  userId: string
  sessionId: string
  agentId?: string
  knowledge?: string[]
  images?: string[]
  signal?: AbortSignal
  onToolCall?: (event: ToolCallEvent) => void
  onThinking?: () => void
  onRagVerification?: (info: unknown) => void
}

interface UseStreamingPipelineReturn {
  sendMessage: (content: string) => Promise<void>
  regenerate: (fromMessageId?: string) => Promise<void>
  stop: () => void
  isStreaming: boolean
  streamingMessageId: string | null
  toolEvents: ToolCallEvent[]
}

/**
 * Token accumulator — buffers tokens and flushes via requestAnimationFrame.
 * Only touches the streaming message (O(1) splice, not O(n) map).
 */
function createTokenAccumulator(
  getMessages: () => ChatMessage[],
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
) {
  const buf: { id: string; text: string }[] = []
  let rafId = 0
  let pending = false

  function flush() {
    pending = false
    if (buf.length === 0) return
    const batch = buf.splice(0)
    const byId = new Map<string, string>()
    for (const { id, text } of batch) {
      byId.set(id, (byId.get(id) || '') + text)
    }
    setMessages(prev => {
      const lastIdx = prev.length - 1
      if (lastIdx < 0) return prev
      const last = prev[lastIdx]
      const delta = byId.get(last.id)
      if (!delta) return prev
      const content = last.content === 'Thinking...' ? '' : last.content
      return [...prev.slice(0, lastIdx), { ...last, content: content + delta }]
    })
  }

  function push(id: string, text: string) {
    buf.push({ id, text })
    if (!pending) {
      pending = true
      rafId = requestAnimationFrame(flush)
    }
  }

  function cancel() {
    if (rafId) cancelAnimationFrame(rafId)
    pending = false
    // Flush remaining
    if (buf.length > 0) flush()
  }

  function reset() {
    cancel()
    buf.length = 0
  }

  return { push, cancel, reset }
}

/**
 * Thinking capture — captures reasoning tokens between <think> tags.
 */
function createThinkingCapture(
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
  messageId: string,
) {
  let buffer = ''
  let capturing = false

  function process(token: string) {
    // Check for think tags
    if (token.includes('<think>')) {
      capturing = true
      buffer += token.split('<think>')[1] || ''
      return { blocked: true }
    }
    if (token.includes('</think>')) {
      const after = token.split('</think>')[1] || ''
      capturing = false
      setMessages(prev => prev.map(m =>
        m.id === messageId ? { ...m, reasoning: buffer } : m
      ))
      buffer = ''
      return { blocked: true, remainder: after }
    }
    if (capturing) {
      buffer += token
      return { blocked: true }
    }
    return { blocked: false }
  }

  return { process }
}

export function useStreamingPipeline({
  messages,
  setMessages,
  model,
  temperature,
  maxTokens,
  systemPrompt,
  userId,
  sessionId,
  agentId,
  knowledge,
  images,
  signal,
  onToolCall,
  onThinking,
  onRagVerification,
}: StreamingPipelineOptions): UseStreamingPipelineReturn {
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [toolEvents, setToolEvents] = useState<ToolCallEvent[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const accumulatorRef = useRef<ReturnType<typeof createTokenAccumulator> | null>(null)
  const thinkingRef = useRef<ReturnType<typeof createThinkingCapture> | null>(null)
  const messagesRef = useRef<ChatMessage[]>(messages)
  messagesRef.current = messages

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      accumulatorRef.current?.reset()
    }
  }, [])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    accumulatorRef.current?.cancel()
    setIsStreaming(false)
    setStreamingMessageId(null)
    setToolEvents([])
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isStreaming) return

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
      images: images?.length ? images.map((dataUrl, i) => ({
        id: `img-${Date.now()}-${i}`,
        dataUrl,
        name: `upload-${i}`,
      })) : undefined,
    }

    const assistantMsg: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsStreaming(true)
    setStreamingMessageId(assistantMsg.id)
    setToolEvents([])

    // Create accumulators
    const acc = createTokenAccumulator(() => messagesRef.current, setMessages)
    accumulatorRef.current = acc
    thinkingRef.current = createThinkingCapture(setMessages, assistantMsg.id)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChatResponse({
        messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })),
        model,
        systemPrompt,
        maxTokens,
        temperature,
        userId,
        sessionId,
        agentId,
        knowledge,
        images,
        signal: controller.signal,
        onToken: (token) => {
          // Check thinking capture
          const result = thinkingRef.current?.process(token)
          if (result?.blocked) {
            if (result.remainder) acc.push(assistantMsg.id, result.remainder)
            return
          }
          acc.push(assistantMsg.id, token)
        },
        onToolCall: (event) => {
          setToolEvents(prev => [...prev, event])
          onToolCall?.(event)
        },
        onThinking: () => {
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id ? { ...m, content: 'Thinking...' } : m
          ))
          onThinking?.()
        },
        onRagVerification: (info) => {
          onRagVerification?.(info)
        },
        onComplete: () => {
          acc.cancel()
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id && !m.content
              ? { ...m, content: '(empty response)' }
              : m
          ))
        },
        onError: (status, text) => {
          acc.cancel()
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id
              ? { ...m, content: m.content || `(error: ${text || 'Stream failed'})`, isError: true }
              : m
          ))
        },
      })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      _log.error('Stream error', { error: String(err) })
      setMessages(prev => prev.map(m =>
        m.id === assistantMsg.id
          ? { ...m, content: `(error: ${String(err)})`, isError: true }
          : m
      ))
    } finally {
      acc.cancel()
      setIsStreaming(false)
      setStreamingMessageId(null)
      abortRef.current = null
      accumulatorRef.current = null
      thinkingRef.current = null
    }
  }, [messages, model, temperature, maxTokens, systemPrompt, userId, sessionId, agentId, knowledge, images, isStreaming, setMessages, onToolCall, onThinking, onRagVerification])

  const regenerate = useCallback(async (fromMessageId?: string) => {
    const currentMessages = messages
    if (currentMessages.length < 2) return

    const targetIdx = fromMessageId
      ? currentMessages.findIndex(m => m.id === fromMessageId)
      : currentMessages.findLastIndex(m => m.role === 'assistant')

    if (targetIdx <= 0) return

    const truncated = currentMessages.slice(0, targetIdx)
    const target = currentMessages[targetIdx]

    // Reset the target message
    setMessages(prev => [
      ...prev.slice(0, targetIdx),
      { ...target, content: '', isError: false, reasoning: undefined },
    ])

    const assistantMsg: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    setMessages(prev => [...prev.slice(0, targetIdx), { ...target, content: '' }, assistantMsg])
    setIsStreaming(true)
    setStreamingMessageId(assistantMsg.id)
    setToolEvents([])

    const acc = createTokenAccumulator(() => messagesRef.current, setMessages)
    accumulatorRef.current = acc
    thinkingRef.current = createThinkingCapture(setMessages, assistantMsg.id)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChatResponse({
        messages: truncated.map(m => ({ role: m.role, content: m.content })),
        model,
        systemPrompt,
        maxTokens,
        temperature,
        userId,
        sessionId,
        agentId,
        knowledge,
        signal: controller.signal,
        onToken: (token) => {
          const result = thinkingRef.current?.process(token)
          if (result?.blocked) {
            if (result.remainder) acc.push(assistantMsg.id, result.remainder)
            return
          }
          acc.push(assistantMsg.id, token)
        },
        onToolCall: (event) => setToolEvents(prev => [...prev, event]),
        onThinking: () => {
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id ? { ...m, content: 'Thinking...' } : m
          ))
        },
        onComplete: () => {
          acc.cancel()
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id && !m.content
              ? { ...m, content: '(empty response)' }
              : m
          ))
        },
        onError: (status, text) => {
          acc.cancel()
          setMessages(prev => prev.map(m =>
            m.id === assistantMsg.id
              ? { ...m, content: m.content || `(error: ${text || 'Stream failed'})`, isError: true }
              : m
          ))
        },
      })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      setMessages(prev => prev.map(m =>
        m.id === assistantMsg.id
          ? { ...m, content: `(error: ${String(err)})`, isError: true }
          : m
      ))
    } finally {
      acc.cancel()
      setIsStreaming(false)
      setStreamingMessageId(null)
      abortRef.current = null
      accumulatorRef.current = null
      thinkingRef.current = null
    }
  }, [messages, model, temperature, maxTokens, systemPrompt, userId, sessionId, agentId, knowledge, setMessages])

  return {
    sendMessage,
    regenerate,
    stop,
    isStreaming,
    streamingMessageId,
    toolEvents,
  }
}
