'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChatResponse, type ToolCallEvent } from '@/lib/stream-chat-response'
import {
  cleanStreamedContent, stripAssistantPrefix, getOrCreateUserId,
  generateSessionId, CURRENT_SESSION_KEY, buildLocalPrompt,
  exportConversationAsMarkdown, copyConversationAsMarkdown,
  type ChatMessage, type ImageAttachment, type ChatSession,
} from '@/lib/chat-utils'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'
import { getErrorInfo } from '@/features/chat/components/feedback/ErrorBanner'
import { chatController } from '@/lib/chat-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { multimodalController, type MultimodalCapabilities } from '@/lib/multimodal-controller'
import { chatDB } from '@/lib/db'
import { useErrorStore } from '@/lib/error-store'
import type { SoulNetWebGPU, SoulTransformerWebGPU } from '@/lib/soulnet-webgpu'
import type { AgentDef } from '@/lib/agents'
import type { Soul } from '@/lib/souls-controller'
import { useAppStore, getKnowledgeContext } from '@/lib/store'
import { publishMemoryEvent } from '@/lib/memory-events'
import { useChatSessions } from './useChatSessions'

const _log = logger.child('chat-messages')

interface ChatMessagesConfig {
  model: string
  temperature: number
  maxTokens: number
  currentSoul: Soul | null
  currentAgent: AgentDef | null
  useLocalEngine: boolean
  engineRef: React.MutableRefObject<SoulNetWebGPU | SoulTransformerWebGPU | null>
  engineLoadingRef: React.MutableRefObject<boolean>
  initLocalEngine: () => Promise<boolean>
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
  recordFeedback: (params: {
    userMessage: string
    assistantResponse: string
    rating: 'thumbs_up' | 'thumbs_down'
    conversationId?: string
    qualityScore?: number
    userId?: string
  }) => Promise<boolean>
  fetchStats: () => void
  fetchAdapterStats: () => void
  onVisionUpdate: (caps: MultimodalCapabilities | null, history: string[], vocab: number | undefined) => void
  onKnowledgeUpdate: (ctx: { count: number; context: string }) => void
  customSystemPrompt?: string
}

export function useChatMessages(config: ChatMessagesConfig) {
  const {
    model, temperature, maxTokens, currentSoul,
    currentAgent, useLocalEngine, engineRef, engineLoadingRef,
    initLocalEngine, showToast, recordFeedback,
    onVisionUpdate, onKnowledgeUpdate, customSystemPrompt,
  } = config

  // ── Core state ───────────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionLoading, setSessionLoading] = useState(false)
  const [images, setImages] = useState<ImageAttachment[]>([])
  const [sessionSaved, setSessionSaved] = useState(false)
  const [currentError, setCurrentError] = useState<ReturnType<typeof getErrorInfo> | null>(null)
  const [contextLayers, setContextLayers] = useState<Array<{ type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'; label: string; detail?: string }>>([])
  const [toolEvents, setToolEvents] = useState<ToolCallEvent[]>([])
  const [ragVerification, setRagVerification] = useState<{
    confidence: number
    is_verified: boolean
    hallucination_rate: number
    citations: string
    grounded_claims: number
    hallucinated_claims: number
  } | null>(null)
  const [pendingToolApproval, setPendingToolApproval] = useState<{
    toolName: string
    args?: Record<string, unknown>
  } | null>(null)

  // ── Message selection for bulk operations ──────────────────────────────────
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set())
  const [selectionMode, setSelectionMode] = useState(false)

  // ── Refs for callback access (read-only, never mutated inside setState) ──
  const messagesRef = useRef<ChatMessage[]>([])
  const loadingRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const lastSaveRef = useRef<number>(0)
  const sendMessageRef = useRef<(overrideText?: string) => Promise<void>>(async () => {})
  const handleRegenerateRef = useRef<(() => Promise<void>) | null>(null)
  const newChatRef = useRef<(() => void) | null>(null)
  const draftTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Session operations (delegated) ──────────────────────────────────────
  const sessions = useChatSessions({
    setMessages, setInput, setSessionSaved, setSessionLoading,
    sessionIdRef, messagesRef, showToast,
  })

  // ── Token accumulator for streaming perf ─────────────────────────────────
  // Buffers tokens in a ref and flushes to setMessages every FLUSH_MS.
  // Avoids O(n) array copy per token — flushes at most ~60x/sec.
  const tokenBufRef = useRef<{ id: string; text: string }[]>([])
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const FLUSH_MS = 16

  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions

  const flushTokens = useCallback(() => {
    flushTimerRef.current = null
    const buf = tokenBufRef.current
    if (buf.length === 0) return
    tokenBufRef.current = []
    const byId = new Map<string, string>()
    for (const { id, text } of buf) {
      byId.set(id, (byId.get(id) || '') + text)
    }
    setMessages(prev => {
      const updated = prev.map(m => {
        const delta = byId.get(m.id)
        if (!delta) return m
        const content = m.content === 'Thinking...' ? '' : m.content
        return { ...m, content: content + delta }
      })
      const now = Date.now()
      if (now - lastSaveRef.current > 500) {
        lastSaveRef.current = now
        sessionsRef.current.saveSessionToStorage(updated, sessionIdRef.current).catch((e) => logger.warning('Could not session save', { error: e }))
      }
      return updated
    })
  }, [])

  const scheduleFlush = useCallback(() => {
    if (flushTimerRef.current) return
    flushTimerRef.current = setTimeout(flushTokens, FLUSH_MS)
  }, [flushTokens])

  // Cleanup on unmount — flush remaining tokens before clearing
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current)
      // Flush any remaining tokens to avoid data loss
      const buf = tokenBufRef.current
      if (buf.length > 0) {
        tokenBufRef.current = []
        const byId = new Map<string, string>()
        for (const { id, text } of buf) {
          byId.set(id, (byId.get(id) || '') + text)
        }
        setMessages(prev => prev.map(m => {
          const delta = byId.get(m.id)
          if (!delta) return m
          const content = m.content === 'Thinking...' ? '' : m.content
          return { ...m, content: content + delta }
        }))
      }
    }
  }, [])

  // Keep messagesRef in sync via effect (single source of truth is `messages` state)
  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const storeSessionContext = useCallback(async (sessionId: string, msgs: ChatMessage[]) => {
    try {
      await chatController.saveSessionContext(sessionId, msgs.map(m => ({ role: m.role, content: m.content })))
    } catch (err) {
      _log.warning('Failed to store session context', { sessionId })
    }
  }, [])

  const newChat = useCallback(() => {
    setMessages([])
    setInput('')
    setSessionSaved(false)
    const newId = generateSessionId()
    sessionIdRef.current = newId
    chatDB.setKV(CURRENT_SESSION_KEY, newId)
    chatDB.deleteDraft(newId)
    showToast('New chat started')
  }, [showToast])
  newChatRef.current = newChat

  // ── Regenerate ────────────────────────────────────────────────────────────
  const handleRegenerate = useCallback(async (fromMessageId?: string) => {
    const currentMessages = messagesRef.current
    if (currentMessages.length < 2) return
    const targetIdx = fromMessageId
      ? currentMessages.findIndex(m => m.id === fromMessageId)
      : currentMessages.findLastIndex(m => m.role === 'assistant')
    if (targetIdx <= 0) return
    const contextMessages = currentMessages.slice(0, targetIdx + 1)
    const assistantId = currentMessages[targetIdx].id

    // If regenerating from a non-last message, truncate the conversation after this point
    if (targetIdx < currentMessages.length - 1) {
      setMessages(prev => prev.slice(0, targetIdx).concat({
        ...currentMessages[targetIdx],
        content: '',
        timestamp: new Date(),
      }))
    } else {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId ? { ...msg, content: '', timestamp: new Date() } : msg
      ))
    }

    setLoading(true)
    try {
      const appState = useAppStore.getState()
      const customContext = appState.settings.customContext
      const parts: string[] = []
      if (customSystemPrompt) parts.push(`[System Override]\n${customSystemPrompt}`)
      if (customContext) parts.push(`[Custom Instructions]\n${customContext}`)
      if (currentSoul) {
        parts.push(`[Personality: ${currentSoul.name}]`)
        if (currentSoul.description) parts.push(currentSoul.description)
        if (currentSoul.traits && currentSoul.traits.length > 0) {
          parts.push(`Traits: ${currentSoul.traits.join(', ')}`)
        }
      }
      const systemPrompt = parts.join('\n\n')
      const knowledgeFacts = appState.injectedKnowledge.map((k: { content: string }) => k.content)

      await streamChatResponse({
        messages: contextMessages.map(m => ({ role: m.role, content: m.content })),
        model, systemPrompt, maxTokens, temperature,
        userId: userIdRef.current, sessionId: sessionIdRef.current,
        signal: loadingRef.current?.signal,
        agentId: currentAgent?.id || undefined,
        knowledge: knowledgeFacts.length > 0 ? knowledgeFacts : undefined,
        onToken: (token: string) => {
          tokenBufRef.current.push({ id: assistantId, text: token })
          scheduleFlush()
        },
        onComplete: () => {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId ? { ...msg, content: msg.content || '(empty response)' } : msg
          ))
        },
        onError: (status, text) => {
          setCurrentError(getErrorInfo(status, text))
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
              : msg
          ))
        },
      })
    } catch (err) {
      _log.error('Regenerate error', { exception: String(err) })
      setCurrentError(getErrorInfo(0, extractErrorMessage(err, 'Network error')))
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
          : msg
      ))
    } finally {
      setLoading(false)
      storeSessionContext(sessionIdRef.current, messagesRef.current)
    }
  }, [showToast, storeSessionContext, scheduleFlush, model, temperature, maxTokens, currentSoul, currentAgent, customSystemPrompt])
  handleRegenerateRef.current = handleRegenerate

  // ── Regenerate with Options ──────────────────────────────────────────────
  const handleRegenerateWithOptions = useCallback(async (fromMessageId: string, options: { temperature?: number; maxTokens?: number }) => {
    const currentMessages = messagesRef.current
    if (currentMessages.length < 2) return
    const targetIdx = currentMessages.findIndex(m => m.id === fromMessageId)
    if (targetIdx <= 0) return
    const contextMessages = currentMessages.slice(0, targetIdx + 1)
    const assistantId = currentMessages[targetIdx].id
    const overrideTemp = options.temperature ?? temperature

    // Truncate conversation after this point if regenerating from a non-last message
    if (targetIdx < currentMessages.length - 1) {
      setMessages(prev => prev.slice(0, targetIdx).concat({
        ...currentMessages[targetIdx],
        content: '',
        timestamp: new Date(),
      }))
    } else {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId ? { ...msg, content: '', timestamp: new Date() } : msg
      ))
    }

    setLoading(true)
    try {
      const appState = useAppStore.getState()
      const customContext = appState.settings.customContext
      const parts: string[] = []
      if (customSystemPrompt) parts.push(`[System Override]\n${customSystemPrompt}`)
      if (customContext) parts.push(`[Custom Instructions]\n${customContext}`)
      if (currentSoul) {
        parts.push(`[Personality: ${currentSoul.name}]`)
        if (currentSoul.description) parts.push(currentSoul.description)
        if (currentSoul.traits && currentSoul.traits.length > 0) {
          parts.push(`Traits: ${currentSoul.traits.join(', ')}`)
        }
      }
      const systemPrompt = parts.join('\n\n')

      await streamChatResponse({
        messages: contextMessages.map(m => ({ role: m.role, content: m.content })),
        model, systemPrompt, maxTokens, temperature: overrideTemp,
        userId: userIdRef.current, sessionId: sessionIdRef.current,
        signal: loadingRef.current?.signal,
        onToken: (token: string) => {
          tokenBufRef.current.push({ id: assistantId, text: token })
          scheduleFlush()
        },
        onComplete: () => {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId ? { ...msg, content: msg.content || '(empty response)' } : msg
          ))
        },
        onError: (_status, text) => {
          setCurrentError(getErrorInfo(0, text || 'Stream error'))
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
              : msg
          ))
        },
      })
    } catch (err) {
      setCurrentError(getErrorInfo(0, extractErrorMessage(err, 'Network error')))
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
          : msg
      ))
    } finally {
      setLoading(false)
      storeSessionContext(sessionIdRef.current, messagesRef.current)
    }
  }, [model, temperature, maxTokens, currentSoul, customSystemPrompt, showToast, storeSessionContext, scheduleFlush])

  // ── Feedback ──────────────────────────────────────────────────────────────
  const handleFeedback = useCallback(async (messageId: string, rating: 'thumbs_up' | 'thumbs_down') => {
    const allMsgs = messagesRef.current
    const msgIdx = allMsgs.findIndex(m => m.id === messageId)
    const assistantMsg = allMsgs.find(m => m.id === messageId)
    const userMsg = msgIdx > 0 ? allMsgs[msgIdx - 1] : null
    const success = await recordFeedback({
      userMessage: userMsg?.content || '', assistantResponse: assistantMsg?.content || '',
      rating, conversationId: sessionIdRef.current, userId: userIdRef.current,
    })
    showToast(success ? 'Thanks for the feedback!' : 'Could not submit feedback', success ? 'success' : 'error')
  }, [showToast, recordFeedback])

  const handleThumbsUp = useCallback((messageId: string) => {
    return handleFeedback(messageId, 'thumbs_up')
  }, [handleFeedback])

  const handleThumbsDown = useCallback((messageId: string) => {
    return handleFeedback(messageId, 'thumbs_down')
  }, [handleFeedback])

  // ── Edit / Retry ──────────────────────────────────────────────────────────
  const handleEditMessage = useCallback((messageId: string, newContent: string) => {
    const allMsgs = messagesRef.current
    const msgIndex = allMsgs.findIndex(m => m.id === messageId)
    if (msgIndex === -1) return
    const sliced = allMsgs.slice(0, msgIndex)
    setMessages(sliced)
    setLoading(false)
    setCurrentError(null)
    setTimeout(() => sendMessageRef.current(newContent), 0)
  }, [])

  const handleRetry = useCallback(() => {
    setCurrentError(null)
    const lastUser = messagesRef.current.findLast(m => m.role === 'user')
    if (lastUser?.content) {
      sendMessageRef.current(lastUser.content)
    } else if (input.trim()) {
      sendMessageRef.current()
    }
  }, [input])

  // ── Image handling ────────────────────────────────────────────────────────
  const handleAddImage = useCallback((dataUrl: string) => {
    const newImage: ImageAttachment = {
      id: crypto.randomUUID(), dataUrl, name: `image-${Date.now()}.png`,
    }
    setImages(prev => [...prev, newImage])
    multimodalController.trainImage(dataUrl, newImage.name).then(res => {
      logger.debug('Vision trained on uploaded image', { caption: res.caption })
      multimodalController.getCapabilities().then(caps => {
        multimodalController.getTrainingReport().then(r => {
          onVisionUpdate(caps, r.caption_history || [], r.vocab_size)
        }).catch(err => _log.warning('Could not vision report', { error: String(err) }))
      }).catch(err => _log.warning('Could not vision capabilities', { error: String(err) }))
    }).catch(err => _log.warning('Could not image training', { error: String(err) }))
  }, [onVisionUpdate])

  const handleRemoveImage = useCallback((id: string) => {
    setImages(prev => prev.filter(img => img.id !== id))
  }, [])

  // ── Copy / Export ─────────────────────────────────────────────────────────
  const handleCopy = useCallback(() => {
    showToast('Copied to clipboard')
  }, [showToast])

  const handleExportMarkdown = useCallback(() => {
    exportConversationAsMarkdown(messagesRef.current)
  }, [])

  const handleCopyMarkdown = useCallback(async () => {
    const ok = await copyConversationAsMarkdown(messagesRef.current)
    showToast(ok ? 'Copied to clipboard' : 'Could not copy', ok ? 'success' : 'error')
  }, [showToast])

  // ── Reactions ──────────────────────────────────────────────────────────
  const handleReact = useCallback((messageId: string, emoji: string) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id !== messageId) return msg
      const reactions = { ...msg.reactions }
      reactions[emoji] = (reactions[emoji] || 0) + 1
      return { ...msg, reactions }
    }))
  }, [])

  // ── Pin/Unpin Messages ──────────────────────────────────────────────────
  const handlePin = useCallback((messageId: string) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id !== messageId) return msg
      return { ...msg, pinned: !msg.pinned }
    }))
  }, [])

  // ── Message Selection ──────────────────────────────────────────────────
  const toggleSelectionMode = useCallback(() => {
    setSelectionMode(prev => !prev)
    setSelectedMessageIds(new Set())
  }, [])

  const toggleMessageSelection = useCallback((messageId: string) => {
    setSelectedMessageIds(prev => {
      const next = new Set(prev)
      if (next.has(messageId)) {
        next.delete(messageId)
      } else {
        next.add(messageId)
      }
      return next
    })
  }, [])

  const selectAllMessages = useCallback(() => {
    setSelectedMessageIds(new Set(messages.filter(m => !m.isError).map(m => m.id)))
  }, [messages])

  const clearSelection = useCallback(() => {
    setSelectedMessageIds(new Set())
  }, [])

  const deleteSelectedMessages = useCallback(() => {
    setMessages(prev => prev.filter(msg => !selectedMessageIds.has(msg.id)))
    setSelectedMessageIds(new Set())
    setSelectionMode(false)
  }, [selectedMessageIds])

  // ── Suggestion click ──────────────────────────────────────────────────────
  const handleSuggestionClick = useCallback((text: string) => {
    setInput(text)
    sendMessageRef.current(text)
  }, [])

  // ── Send Message (core — stable reference) ───────────────────────────────
  const sendMessage = useCallback(async (overrideText?: string) => {
    const text = overrideText ?? input
    if ((!text.trim() && images.length === 0) || loading) return
    const userImages = [...images]

    const appState = useAppStore.getState()
    const customContext = appState.settings.customContext

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(), role: 'user', content: text.trim(),
      timestamp: new Date(),
      images: userImages.length > 0 ? userImages : undefined,
    }
    const assistantId = crypto.randomUUID()
    const assistantMessage: ChatMessage = {
      id: assistantId, role: 'assistant', content: '', timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])
    setInput('')
    chatDB.deleteDraft(sessionIdRef.current!)
    setImages([])
    setCurrentError(null)
    setToolEvents([])
    setRagVerification(null)
    setContextLayers([])
    setLoading(true)

    const parts: string[] = []
    if (customSystemPrompt) {
      parts.push(`[System Override]\n${customSystemPrompt}`)
    }
    if (customContext) {
      parts.push(`[Custom Instructions]\n${customContext}`)
    }
    if (currentSoul) {
      parts.push(`[Personality: ${currentSoul.name}]`)
      if (currentSoul.description) parts.push(currentSoul.description)
      if (currentSoul.traits && currentSoul.traits.length > 0) {
        parts.push(`Traits: ${currentSoul.traits.join(', ')}`)
      }
    }
    if (currentAgent) {
      parts.push(`[Role: ${currentAgent.name}]`)
      if (currentAgent.description) parts.push(currentAgent.description)
    }
    const systemPrompt = parts.join('\n\n')
    const knowledgeFacts = appState.injectedKnowledge.map((k: { content: string }) => k.content)

    // Build context layers for reasoning panel
    const initialContextLayers: Array<{ type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'; label: string; detail?: string }> = []
    if (currentSoul) {
      initialContextLayers.push({ type: 'soul', label: `Personality: ${currentSoul.name}`, detail: currentSoul.description })
    }
    if (currentAgent) {
      initialContextLayers.push({ type: 'system', label: `Agent: ${currentAgent.name}`, detail: currentAgent.description })
    }
    if (knowledgeFacts.length > 0) {
      initialContextLayers.push({ type: 'knowledge', label: 'Knowledge context', detail: `${knowledgeFacts.length} facts injected` })
    }
    setContextLayers(initialContextLayers)

    const messagesWithNew = [...messagesRef.current, userMessage, assistantMessage]
    sessions.saveSessionToStorage(messagesWithNew, sessionIdRef.current).catch((e) => logger.warning('Could not session save', { error: e }))
    messagesRef.current = messagesWithNew
    loadingRef.current = new AbortController()

    if (useLocalEngine && !engineRef.current && !engineLoadingRef.current) {
      await initLocalEngine()
    }

    try {
      if (useLocalEngine && engineRef.current) {
        const prompt = buildLocalPrompt(messagesWithNew, systemPrompt)
        let hasContent = false
        let assistantContentLen = 0
        const signal = loadingRef.current?.signal
        for await (const token of engineRef.current.generate(prompt, maxTokens, temperature)) {
          if (signal?.aborted) break
          hasContent = true
          let cleanedToken = token
          if (assistantContentLen < 50) {
            cleanedToken = stripAssistantPrefix(cleanedToken)
            cleanedToken = cleanStreamedContent(cleanedToken)
          }
          assistantContentLen += cleanedToken.length
          tokenBufRef.current.push({ id: assistantId, text: cleanedToken })
          scheduleFlush()
        }
        if (!hasContent) {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId ? { ...msg, content: '(empty response)' } : msg
          ))
        }
        setSessionSaved(true)
      } else {
        let assistantContentLen = 0
        let streamComplete = false
        const finalSystemPrompt = systemPrompt
        await streamChatResponse({
          messages: messagesWithNew
            .filter(m => m.content.trim().length > 0 || m.role === 'user')
            .map(m => ({ role: m.role, content: m.content })),
          model, systemPrompt: finalSystemPrompt, maxTokens, temperature,
          userId: userIdRef.current, sessionId: sessionIdRef.current,
          images: userImages.length > 0 ? userImages.map(img => img.dataUrl) : undefined,
          signal: loadingRef.current.signal,
          agentId: currentAgent?.id || undefined,
          knowledge: knowledgeFacts.length > 0 ? knowledgeFacts : undefined,
          onToken: (token: string) => {
            let cleanedToken = token
            if (assistantContentLen < 50) {
              cleanedToken = stripAssistantPrefix(cleanedToken)
              cleanedToken = cleanStreamedContent(cleanedToken)
            }
            assistantContentLen += cleanedToken.length
            tokenBufRef.current.push({ id: assistantId, text: cleanedToken })
            scheduleFlush()
          },
          onComplete: () => {
            streamComplete = true
            setSessionSaved(true)
            flushTokens()
            setLoading(false)
            setMessages(prev => prev.map(m =>
              m.id === assistantId && m.content === 'Thinking...' ? { ...m, content: '' } : m
            ))
          },
          onError: (status: number, text?: string) => {
            flushTokens()
            setCurrentError(getErrorInfo(status, text || 'Stream error'))
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
                : msg
            ))
            setLoading(false)
          },
          onKnowledge: (source: string, count: number) => {
            showToast(`Knowledge: ${count} facts from ${source}`, 'info')
            setContextLayers(prev => [...prev, { type: 'knowledge', label: `Knowledge: ${source}`, detail: `${count} facts` }])
          },
          onMemory: (info) => {
            publishMemoryEvent(info)
            if (info.stored) {
              const list = (info.facts && info.facts.length > 0 ? info.facts : info.fact ? [info.fact] : []) as string[]
              const first = list[0]
              const extra = list.length > 1 ? ` +${list.length - 1} more` : ''
              const shown = first && first.length > 140 ? `${first.slice(0, 140)}…` : first
              showToast(shown ? `Remembered: ${shown}${extra}` : 'New fact saved to memory', 'success')
              setContextLayers(prev => [...prev, { type: 'memory', label: 'Memory updated', detail: first ? `${list.length} fact${list.length > 1 ? 's' : ''} stored` : undefined }])
            }
          },
          onThinking: () => {
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId && !msg.content ? { ...msg, content: 'Thinking...' } : msg
            ))
          },
          onToolCall: (event) => {
            setToolEvents(prev => [...prev, event])
            setContextLayers(prev => [...prev, { type: 'tool', label: `Tool: ${event.tool}`, detail: event.status }])
            if (event.status === 'executing' && event.args) {
              const autoApprove = useAppStore.getState().settings.autoApproveTools
              if (autoApprove) {
                chatController.approveTool(sessionIdRef.current, event.tool, true)
              } else {
                setPendingToolApproval({ toolName: event.tool, args: event.args })
              }
            }
          },
          onRagVerification: (info) => {
            setRagVerification(info)
            setContextLayers(prev => [...prev, { type: 'rag', label: 'RAG verification', detail: `${(info.confidence * 100).toFixed(0)}% confidence` }])
          },
          onControl: (event) => {
            if (event.action === 'cancelled') {
              setMessages(prev => prev.map(msg =>
                msg.id === assistantId
                  ? { ...msg, content: msg.content || '(cancelled)', isError: true }
                  : msg
              ))
            } else if (event.action === 'context') {
              setContextLayers(prev => [...prev, { type: 'system', label: 'Context injected', detail: event.context }])
            }
          },
        })
        if (streamComplete) {
          storeSessionContext(sessionIdRef.current, messagesRef.current).catch((e) => logger.warning('Could not session context store', { error: e }))
          knowledgeController.context().then(res => {
            onKnowledgeUpdate({ count: res.count, context: res.context })
          }).catch((e) => logger.debug('Could not search query', e))
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        logger.debug('Stream aborted by user')
      } else {
        setCurrentError(getErrorInfo(0, extractErrorMessage(err, 'Network error')))
        setMessages(prev => prev.map(msg =>
          msg.id === assistantId
            ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
            : msg
        ))
      }
    } finally {
      setLoading(false)
      loadingRef.current = null
    }
  }, [
    input, images, loading, model, temperature, maxTokens, currentSoul, currentAgent,
    useLocalEngine, engineRef, engineLoadingRef, initLocalEngine,
    showToast, onKnowledgeUpdate, storeSessionContext, sessions,
    customSystemPrompt, flushTokens, scheduleFlush,
  ])
  sendMessageRef.current = sendMessage

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    chatDB.getKV<string>(CURRENT_SESSION_KEY).then(existing => {
      if (existing) {
        sessionIdRef.current = existing
      } else {
        const fresh = generateSessionId()
        sessionIdRef.current = fresh
        chatDB.setKV(CURRENT_SESSION_KEY, fresh)
      }
    })
    getOrCreateUserId().then(id => { userIdRef.current = id })
    chatDB.getDraft(sessionIdRef.current).then(savedDraft => {
      if (savedDraft) setInput(savedDraft)
    })
  }, [])

  useEffect(() => {
    chatDB.getKV<string>(CURRENT_SESSION_KEY).then(currentId => {
      if (currentId) sessionsRef.current.loadSession(currentId)
    })
  }, [])

  // Session save: ref-based debounce (doesn't recreate on messages change)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (messagesRef.current.length > 0 && sessionSaved) {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = setTimeout(() => {
        const sid = sessionIdRef.current || generateSessionId()
        sessionsRef.current.saveSessionToStorage(messagesRef.current, sid)
      }, 1000)
      return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current) }
    }
  }, [sessionSaved])

  useEffect(() => {
    let ignore = false
    const handler = async () => {
      try {
        const backendSessions = await (await import('@/lib/session-controller')).sessionController.list()
        if (ignore) return
        const localSessions = await chatDB.loadSessions()
        if (ignore) return
        const merged: ChatSession[] = [
          ...(Array.isArray(backendSessions) ? backendSessions : []).map((s): ChatSession => ({
            id: s.id, name: s.name || `Chat ${s.id}`,
            messages: (s.messages || []).map(m => ({
              id: m.id || `msg_${Date.now()}`, role: (m.role || 'user') as 'user' | 'assistant',
              content: m.content || '', timestamp: new Date(m.timestamp || Date.now()),
            })),
            createdAt: s.created_at, updatedAt: s.updated_at,
            synced: true, starred: s.starred ?? false, pinned: s.pinned ?? false,
          })),
          ...localSessions.filter(l => !Array.isArray(backendSessions) || !backendSessions.find((b: { id: string }) => b.id === l.id)),
        ]
        if (ignore) return
        sessionsRef.current.setSessions(merged)
      } catch (err) {
        if (ignore) return
        useErrorStore.getState().addError(err, { source: 'Chat Sessions' })
        const localSessions = await chatDB.loadSessions()
        if (ignore) return
        sessionsRef.current.setSessions(localSessions)
      }
    }
    handler()
    return () => { ignore = true }
  }, [])

  useEffect(() => {
    const handler = (e: CustomEvent) => {
      const { dataUrl, prompt } = e.detail
      setMessages(prev => [...prev, {
        id: `img-${Date.now()}`, role: 'user',
        content: `[Generate image: ${prompt}]`, timestamp: new Date(),
        images: [{ id: `gen-${Date.now()}`, dataUrl, name: 'generated.png' }],
      }])
      showToast('Image generated in conversation', 'info')
    }
    window.addEventListener('insert-generated-image', handler as EventListener)
    return () => window.removeEventListener('insert-generated-image', handler as EventListener)
  }, [showToast])

  useEffect(() => {
    const handler = (e: CustomEvent) => {
      if (e.detail?.text) sendMessageRef.current(e.detail.text)
    }
    window.addEventListener('send-text', handler as EventListener)
    return () => window.removeEventListener('send-text', handler as EventListener)
  }, [])

  useEffect(() => {
    if (draftTimerRef.current) clearTimeout(draftTimerRef.current)
    draftTimerRef.current = setTimeout(() => {
      const sid = sessionIdRef.current
      if (sid && input) {
        chatDB.saveDraft(sid, input)
      }
    }, 500)
    return () => {
      if (draftTimerRef.current) clearTimeout(draftTimerRef.current)
    }
  }, [input])

  return {
    messages, setMessages,
    input, setInput,
    loading, setLoading,
    sessionLoading,
    images,
    sessionSaved,
    currentError, setCurrentError,
    toolEvents,
    ragVerification,
    contextLayers,
    messagesRef,
    loadingRef,
    sessionIdRef,
    userIdRef,
    handleRegenerateRef,
    newChatRef,
    sendMessageRef,
    sendMessage,
    newChat,
    loadSession: sessions.loadSession,
    deleteSession: sessions.deleteSession,
    starSession: sessions.starSession,
    pinSession: sessions.pinSession,
    archiveSession: sessions.archiveSession,
    archivedCount: sessions.archivedCount,
    renameSession: sessions.renameSession,
    duplicateSession: sessions.duplicateSession,
    handleRegenerate,
    handleRegenerateWithOptions,
    handleThumbsUp, handleThumbsDown,
    handleEditMessage,
    handleAddImage, handleRemoveImage,
    handleCopy,
    handleRetry,
    handleSuggestionClick,
    handleExportMarkdown,
    handleCopyMarkdown,
    handleReact,
    handlePin,
    selectedMessageIds,
    selectionMode,
    toggleSelectionMode,
    toggleMessageSelection,
    selectAllMessages,
    clearSelection,
    deleteSelectedMessages,
    sidebarConversations: sessions.sidebarConversations,
    cancelStream: useCallback(() => chatController.cancelStream(sessionIdRef.current), []),
    approveTool: useCallback((toolName: string, approved: boolean) =>
      chatController.approveTool(sessionIdRef.current, toolName, approved), []),
    injectContext: useCallback((context: string) =>
      chatController.injectContext(sessionIdRef.current, context), []),
    pendingToolApproval,
    handleToolApproval: useCallback((approved: boolean) => {
      if (pendingToolApproval) {
        chatController.approveTool(sessionIdRef.current, pendingToolApproval.toolName, approved)
        setPendingToolApproval(null)
      }
    }, [pendingToolApproval]),
  }
}
