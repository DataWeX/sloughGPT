'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChatResponse } from '@/lib/stream-chat-response'
import {
  cleanStreamedContent, stripAssistantPrefix, getOrCreateUserId,
  generateSessionId, CURRENT_SESSION_KEY, buildLocalPrompt,
  exportConversationAsMarkdown, copyConversationAsMarkdown,
  type ChatMessage, type ImageAttachment, type ChatSession,
} from '@/lib/chat-utils'
import { logger } from '@/lib/dev-log'

const _log = logger.child('chat-messages')
import { getErrorInfo } from '@/components/chat/ErrorBanner'
import { chatController } from '@/lib/chat-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { multimodalController } from '@/lib/multimodal-controller'
import { chatDB } from '@/lib/db'
import { useErrorStore } from '@/lib/error-store'
import { devDebug } from '@/lib/dev-log'
import type { SoulNetWebGPU, SoulTransformerWebGPU } from '@/lib/soulnet-webgpu'
import type { AgentDef } from '@/lib/agents'
import type { Soul } from '@/lib/souls-controller'
import type { MultimodalCapabilities } from '@/lib/multimodal-controller'
import type { ToolCallEvent } from '@/lib/stream-chat-response'
import { useAppStore, getKnowledgeContext } from '@/lib/store'
import { useChatSessions } from './useChatSessions'


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
  showToast: (message: string, type?: string) => void
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
  const [toolEvents, setToolEvents] = useState<ToolCallEvent[]>([])

  // ── Refs for callback access (read-only, never mutated inside setState) ──
  const messagesRef = useRef<ChatMessage[]>([])
  const loadingRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const lastSaveRef = useRef<number>(0)
  const sendMessageRef = useRef<(overrideText?: string) => Promise<void>>(async () => {})
  const handleRegenerateRef = useRef<() => Promise<void>>(null as unknown as () => Promise<void>)
  const newChatRef = useRef<() => void>(null as unknown as () => void)
  const draftTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Session operations (delegated) ──────────────────────────────────────
  const sessions = useChatSessions({
    setMessages, setInput, setSessionSaved, setSessionLoading,
    sessionIdRef, showToast,
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
        return delta ? { ...m, content: m.content + delta } : m
      })
      const now = Date.now()
      if (now - lastSaveRef.current > 500) {
        lastSaveRef.current = now
        sessionsRef.current.saveSessionToStorage(updated, sessionIdRef.current).catch((e) => logger.warning('Session save failed', { error: e }))
      }
      return updated
    })
  }, [])

  const scheduleFlush = useCallback(() => {
    if (flushTimerRef.current) return
    flushTimerRef.current = setTimeout(flushTokens, FLUSH_MS)
  }, [flushTokens])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current)
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
  const handleRegenerate = useCallback(async () => {
    const currentMessages = messagesRef.current
    if (currentMessages.length < 2) return
    const lastAssistantIdx = currentMessages.findLastIndex(m => m.role === 'assistant')
    if (lastAssistantIdx <= 0) return
    const contextMessages = currentMessages.slice(0, lastAssistantIdx + 1)
    const assistantId = currentMessages[lastAssistantIdx].id
    setMessages(prev => prev.map(msg =>
      msg.id === assistantId ? { ...msg, content: '', timestamp: new Date() } : msg
    ))
    setLoading(true)
    try {
      for await (const data of chatController.regenerateStream(
        sessionIdRef.current,
        contextMessages.map(m => ({ role: m.role, content: m.content }))
      )) {
        if (data.error) { showToast('Failed to regenerate response', 'error'); break }
        if (data.token) {
          tokenBufRef.current.push({ id: assistantId, text: data.token })
          scheduleFlush()
        }
        if (data.done) break
      }
    } catch (err) {
      _log.error('Regenerate error', { exception: String(err) })
    } finally {
      setLoading(false)
      storeSessionContext(sessionIdRef.current, messagesRef.current)
    }
  }, [showToast, storeSessionContext])
  handleRegenerateRef.current = handleRegenerate

  // ── Feedback ──────────────────────────────────────────────────────────────
  const handleThumbsUp = useCallback(async (messageId: string) => {
    const allMsgs = messagesRef.current
    const msgIdx = allMsgs.findIndex(m => m.id === messageId)
    const assistantMsg = allMsgs[msgIdx]
    const userMsg = msgIdx > 0 ? allMsgs[msgIdx - 1] : null
    const success = await recordFeedback({
      userMessage: userMsg?.content || '', assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_up', conversationId: sessionIdRef.current, userId: userIdRef.current,
    })
    showToast(success ? 'Thanks for the feedback!' : 'Failed to submit feedback', success ? 'success' : 'error')
  }, [showToast, recordFeedback])

  const handleThumbsDown = useCallback(async (messageId: string) => {
    const allMsgs = messagesRef.current
    const msgIdx = allMsgs.findIndex(m => m.id === messageId)
    const assistantMsg = allMsgs[msgIdx]
    const userMsg = msgIdx > 0 ? allMsgs[msgIdx - 1] : null
    const success = await recordFeedback({
      userMessage: userMsg?.content || '', assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_down', conversationId: sessionIdRef.current, userId: userIdRef.current,
    })
    showToast(success ? 'Thanks for the feedback!' : 'Failed to submit feedback', success ? 'success' : 'error')
  }, [showToast, recordFeedback])

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
      id: Date.now().toString(), dataUrl, name: `image-${Date.now()}.png`,
    }
    setImages(prev => [...prev, newImage])
    multimodalController.trainImage(dataUrl, newImage.name).then(res => {
      devDebug('Vision trained on uploaded image', res.caption)
      multimodalController.getCapabilities().then(caps => {
        multimodalController.getTrainingReport().then(r => {
          onVisionUpdate(caps, r.caption_history || [], r.vocab_size)
        }).catch(err => _log.warning('Vision report failed', { error: String(err) }))
      }).catch(err => _log.warning('Vision capabilities failed', { error: String(err) }))
    }).catch(err => _log.warning('Image training failed', { error: String(err) }))
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
    showToast(ok ? 'Copied to clipboard' : 'Failed to copy', ok ? 'success' : 'error')
  }, [showToast])

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
      id: Date.now().toString(), role: 'user', content: text.trim(),
      timestamp: new Date(),
      images: userImages.length > 0 ? userImages : undefined,
    }
    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: ChatMessage = {
      id: assistantId, role: 'assistant', content: '', timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])
    setInput('')
    chatDB.deleteDraft(sessionIdRef.current!)
    setImages([])
    setCurrentError(null)
    setToolEvents([])
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
      if (currentAgent.instructions) parts.push(currentAgent.instructions)
    }
    const systemPrompt = parts.join('\n\n')
    const knowledgeCtx = getKnowledgeContext()
    const knowledgeFacts = appState.injectedKnowledge.map((k: { content: string }) => k.content)

    const messagesWithNew = [...messagesRef.current, userMessage, assistantMessage]
    sessions.saveSessionToStorage(messagesWithNew, sessionIdRef.current).catch((e) => logger.warning('Session save failed', { error: e }))
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
        for await (const token of engineRef.current.generate(prompt, maxTokens, temperature)) {
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
        const finalSystemPrompt = knowledgeCtx ? systemPrompt + knowledgeCtx : systemPrompt
        await streamChatResponse({
          messages: messagesWithNew.map(m => ({ role: m.role, content: m.content })),
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
          },
          onThinking: () => {
            setMessages(prev => prev.map(msg =>
              msg.id === assistantId && !msg.content ? { ...msg, content: 'Thinking...' } : msg
            ))
          },
          onToolCall: (event) => {
            setToolEvents(prev => [...prev, event])
          },
        })
        if (streamComplete) {
          storeSessionContext(sessionIdRef.current, messagesRef.current).catch((e) => logger.warning('Session context store failed', { error: e }))
          knowledgeController.context().then(res => {
            onKnowledgeUpdate({ count: res.count, context: res.context })
          }).catch((e) => logger.debug('Search query failed', e))
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        devDebug('Stream aborted by user')
      } else {
        setCurrentError(getErrorInfo(0, err instanceof Error ? err.message : 'Network error'))
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
      if (currentId) sessions.loadSession(currentId)
    })
  }, [])

  // Session save: ref-based debounce (doesn't recreate on messages change)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (messagesRef.current.length > 0 && sessionSaved) {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      saveTimerRef.current = setTimeout(() => {
        const sid = sessionIdRef.current || generateSessionId()
        sessions.saveSessionToStorage(messagesRef.current, sid)
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
        sessions.setSessions(merged)
      } catch (err) {
        if (ignore) return
        useErrorStore.getState().addError(err, { source: 'Chat Sessions' })
        const localSessions = await chatDB.loadSessions()
        if (ignore) return
        sessions.setSessions(localSessions)
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
    handleThumbsUp, handleThumbsDown,
    handleEditMessage,
    handleAddImage, handleRemoveImage,
    handleCopy,
    handleRetry,
    handleSuggestionClick,
    handleExportMarkdown,
    handleCopyMarkdown,
    sidebarConversations: sessions.sidebarConversations,
  }
}
