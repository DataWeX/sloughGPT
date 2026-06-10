'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { streamChatResponse } from '@/lib/stream-chat-response'
import {
  cleanStreamedContent, stripAssistantPrefix, getOrCreateUserId,
  generateSessionId, CURRENT_SESSION_KEY, buildLocalPrompt,
  exportConversationAsMarkdown,
  type ChatMessage, type ImageAttachment, type ChatSession,
} from '@/lib/chat-utils'
import { getErrorInfo } from '@/components/chat/ErrorBanner'
import type { Conversation } from '@/lib/session-controller'
import { chatController } from '@/lib/chat-controller'
import { sessionController } from '@/lib/session-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { multimodalController } from '@/lib/multimodal-controller'
import { chatDB, type ChatSession as DBChatSession } from '@/lib/db'
import { useErrorStore, addGlobalError } from '@/lib/error-store'
import { devDebug } from '@/lib/dev-log'
import type { SoulNetWebGPU, SoulTransformerWebGPU } from '@/lib/soulnet-webgpu'
import type { AgentDef } from '@/lib/agents'
import type { Soul } from '@/lib/souls-controller'
import type { MultimodalCapabilities } from '@/lib/multimodal-controller'

const MAX_STORAGE_MESSAGES = 40
const VOICE_MODE_KEY = 'man_voice_mode'

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
}

export function useChatMessages(config: ChatMessagesConfig) {
  const {
    model, temperature, maxTokens, currentSoul,
    currentAgent, useLocalEngine, engineRef, engineLoadingRef,
    initLocalEngine, showToast, recordFeedback,
    onVisionUpdate, onKnowledgeUpdate,
  } = config

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [images, setImages] = useState<ImageAttachment[]>([])
  const [sessionSaved, setSessionSaved] = useState(false)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentError, setCurrentError] = useState<ReturnType<typeof getErrorInfo> | null>(null)

  const messagesRef = useRef<ChatMessage[]>([])
  const loadingRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const lastSaveRef = useRef<number>(0)
  const sessionCreatedRef = useRef(false)
  const streamStartRef = useRef<number>(0)
  const sendMessageRef = useRef<(overrideText?: string) => Promise<void>>(async () => {})
  const skipMeteredWarningRef = useRef(false)
  const handleRegenerateRef = useRef<() => Promise<void>>(null as unknown as () => Promise<void>)
  const newChatRef = useRef<() => void>(null as unknown as () => void)

  // ── Session / Storage ─────────────────────────────────────────────────────

  const saveSessionToStorage = async (msgs: ChatMessage[], sessionId: string) => {
    const sessionName = msgs[0]?.content?.slice(0, 30) || 'New Chat'
    const storedMsgs = msgs.length > MAX_STORAGE_MESSAGES
      ? msgs.slice(-MAX_STORAGE_MESSAGES)
      : msgs
    const session: DBChatSession = {
      id: sessionId, name: sessionName, messages: storedMsgs,
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
      synced: false, starred: false, pinned: false,
    }
    await chatDB.saveSession(session)
    if (!sessionCreatedRef.current) {
      sessionController.create(sessionName, sessionId).catch(console.error)
      sessionCreatedRef.current = true
    } else {
      sessionController.update(sessionId, { name: sessionName }).catch(err => addGlobalError(err, 'Chat:SessionUpdate'))
    }
    const newLocal = await chatDB.loadSessions()
    setSessions(prev => {
      const merged = [...(prev || [])]
      for (const p of prev || []) {
        if (!merged.find(m => m.id === p.id) && p.id) merged.push(p)
      }
      return merged.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''))
    })
  }

  const saveSession = useCallback(async () => {
    if (messages.length === 0) return
    const sid = sessionIdRef.current || localStorage.getItem(CURRENT_SESSION_KEY) || generateSessionId()
    await saveSessionToStorage(messages, sid)
  }, [messages])

  const storeSessionContext = async (sessionId: string, msgs: ChatMessage[]) => {
    try {
      await chatController.saveSessionContext(sessionId, msgs.map(m => ({ role: m.role, content: m.content })))
    } catch {}
  }

  const loadSession = useCallback(async (sessionId: string) => {
    const session = await chatDB.loadSession(sessionId)
    if (session) {
      const filteredMessages = session.messages.filter(msg => {
        if (msg.role === 'assistant' && !msg.content.trim()) return false
        return true
      })
      try {
        const remoteMsgs = await sessionController.fetchMessages(sessionId)
        const remoteChatMsgs = remoteMsgs.map(m => ({
          id: `remote_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: new Date(),
        }))
        const seen = new Set(remoteChatMsgs.map(m => `${m.role}:${m.content}`))
        const uniqueLocal = filteredMessages.filter(m => !seen.has(`${m.role}:${m.content}`))
        setMessages([...remoteChatMsgs, ...uniqueLocal])
      } catch {
        setMessages(filteredMessages)
      }
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
      const isComplete = filteredMessages.length > 0 &&
        filteredMessages[filteredMessages.length - 1].role === 'assistant'
      setSessionSaved(isComplete)
      if (filteredMessages.length > 0) showToast(`Loaded: ${session.name}`)
    }
  }, [showToast])

  const newChat = useCallback(() => {
    setMessages([])
    setSessionSaved(false)
    const newId = generateSessionId()
    sessionIdRef.current = newId
    sessionCreatedRef.current = false
    localStorage.setItem(CURRENT_SESSION_KEY, newId)
    showToast('New chat started')
  }, [showToast])
  newChatRef.current = newChat

  // ── Session CRUD ──────────────────────────────────────────────────────────

  const deleteSession = useCallback(async (sessionId: string) => {
    await chatDB.deleteSession(sessionId)
    sessionController.delete(sessionId).catch(console.error)
    const newSessions = await chatDB.loadSessions()
    setSessions(newSessions)
    if (localStorage.getItem(CURRENT_SESSION_KEY) === sessionId) {
      setMessages([])
      setSessionSaved(false)
      localStorage.removeItem(CURRENT_SESSION_KEY)
    }
    showToast('Session deleted')
  }, [showToast])

  const starSession = useCallback(async (sessionId: string, starred: boolean) => {
    await chatDB.updateSession(sessionId, { starred })
    sessionController.update(sessionId, { starred }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, starred } : s))
    showToast(starred ? 'Conversation starred' : 'Conversation unstarred')
  }, [showToast])

  const pinSession = useCallback(async (sessionId: string, pinned: boolean) => {
    await chatDB.updateSession(sessionId, { pinned })
    sessionController.update(sessionId, { pinned }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, pinned } : s))
    showToast(pinned ? 'Conversation pinned' : 'Conversation unpinned')
  }, [showToast])

  const renameSession = useCallback(async (sessionId: string, newName: string) => {
    await chatDB.updateSession(sessionId, { name: newName })
    sessionController.update(sessionId, { name: newName }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, name: newName } : s))
    showToast('Conversation renamed')
  }, [showToast])

  const duplicateSession = useCallback(async (sessionId: string) => {
    const session = await chatDB.loadSession(sessionId)
    if (session) {
      const newId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      const newSession = {
        ...session, id: newId, name: `${session.name} (copy)`,
        createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
        starred: false, pinned: false,
      }
      await chatDB.saveSession(newSession)
      const newSessions = await chatDB.loadSessions()
      setSessions(newSessions)
      showToast('Conversation duplicated')
    }
  }, [showToast])

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
        if (data.error) { showToast('Regeneration failed', 'error'); break }
        if (data.token) {
          setMessages(prev => {
            const updated = prev.map(msg =>
              msg.id === assistantId ? { ...msg, content: (msg.content || '') + data.token } : msg
            )
            messagesRef.current = updated
            return updated
          })
        }
        if (data.done) break
      }
    } catch (err) {
      console.error('Regenerate error:', err)
    } finally {
      setLoading(false)
      storeSessionContext(sessionIdRef.current, messagesRef.current)
    }
  }, [showToast])
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
    const keepCount = msgIndex
    const sliced = allMsgs.slice(0, keepCount)
    messagesRef.current = sliced
    setMessages(sliced)
    setLoading(false)
    setCurrentError(null)
    setTimeout(() => sendMessageRef.current(newContent), 0)
  }, [])

  const handleRetry = () => {
    setCurrentError(null)
    const lastUser = messagesRef.current.findLast(m => m.role === 'user')
    if (lastUser?.content) {
      sendMessage(lastUser.content)
    } else if (input.trim()) {
      sendMessage()
    }
  }

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
        }).catch(() => {})
      }).catch(() => {})
    }).catch(() => {})
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

  // ── Suggestion click ──────────────────────────────────────────────────────

  const handleSuggestionClick = useCallback((text: string) => {
    setInput(text)
    sendMessageRef.current(text)
  }, [])

  // ── Send Message (core) ──────────────────────────────────────────────────

  const sendMessage = async (overrideText?: string) => {
    sendMessageRef.current = sendMessage
    const text = overrideText ?? input
    if ((!text.trim() && images.length === 0) || loading) return
    const userImages = [...images]

    let customContext = ''
    try {
      const settingsStored = localStorage.getItem('man_settings')
      if (settingsStored) {
        const settings = JSON.parse(settingsStored)
        customContext = settings.customContext || ''
      }
    } catch {}

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
    setImages([])
    setCurrentError(null)
    setLoading(true)
    streamStartRef.current = Date.now()

    const parts: string[] = []
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

    const messagesWithNew = [...messagesRef.current, userMessage, assistantMessage]
    saveSessionToStorage(messagesWithNew, sessionIdRef.current).catch(console.error)
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
          const now = Date.now()
          const shouldSave = now - lastSaveRef.current > 500
          if (shouldSave) lastSaveRef.current = now
          setMessages(prev => {
            const updated = prev.map(m =>
              m.id === assistantId ? { ...m, content: m.content + cleanedToken } : m
            )
            messagesRef.current = updated
            if (shouldSave) saveSessionToStorage(updated, sessionIdRef.current).catch(console.error)
            return updated
          })
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
        await streamChatResponse({
          messages: messagesWithNew.map(m => ({ role: m.role, content: m.content })),
          model, systemPrompt, maxTokens, temperature,
          userId: userIdRef.current, sessionId: sessionIdRef.current,
          images: userImages.length > 0 ? userImages.map(img => img.dataUrl) : undefined,
          signal: loadingRef.current.signal,
          onToken: (token: string) => {
            let cleanedToken = token
            if (assistantContentLen < 50) {
              cleanedToken = stripAssistantPrefix(cleanedToken)
              cleanedToken = cleanStreamedContent(cleanedToken)
            }
            assistantContentLen += cleanedToken.length
            const now = Date.now()
            const shouldSave = now - lastSaveRef.current > 500
            if (shouldSave) lastSaveRef.current = now
            const needsSave = shouldSave
            setMessages(prev => {
              const updated = prev.map(m => {
                if (m.id !== assistantId) return m
                const content = m.content === 'Thinking...' ? '' : m.content
                return { ...m, content: content + cleanedToken }
              })
              messagesRef.current = updated
              if (needsSave) saveSessionToStorage(updated, sessionIdRef.current).catch(console.error)
              return updated
            })
          },
          onComplete: () => {
            streamComplete = true
            setSessionSaved(true)
            setMessages(prev => prev.map(m =>
              m.id === assistantId && m.content === 'Thinking...' ? { ...m, content: '' } : m
            ))
          },
          onError: (status: number, text?: string) => {
            setCurrentError(getErrorInfo(status, text || 'Stream error'))
            setMessages(prev => prev.filter(msg => msg.id !== assistantId))
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
        })
        if (streamComplete) {
          storeSessionContext(sessionIdRef.current, messagesRef.current).catch(console.error)
          knowledgeController.context().then(res => {
            onKnowledgeUpdate({ count: res.count, context: res.context })
          }).catch(() => {})
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        devDebug('Stream aborted by user')
      } else {
        setCurrentError(getErrorInfo(0, err instanceof Error ? err.message : 'Network error'))
        setMessages(prev => prev.filter(msg => msg.id !== assistantId))
      }
    } finally {
      setLoading(false)
      loadingRef.current = null
    }
  }

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const existing = localStorage.getItem(CURRENT_SESSION_KEY)
    if (existing) {
      sessionIdRef.current = existing
    } else {
      const fresh = generateSessionId()
      sessionIdRef.current = fresh
      localStorage.setItem(CURRENT_SESSION_KEY, fresh)
    }
    userIdRef.current = getOrCreateUserId()
  }, [])

  useEffect(() => {
    const currentId = localStorage.getItem(CURRENT_SESSION_KEY)
    if (currentId) loadSession(currentId)
  }, [])

  useEffect(() => {
    if (messages.length > 0 && sessionSaved) {
      const timeout = setTimeout(saveSession, 1000)
      return () => clearTimeout(timeout)
    }
  }, [messages, saveSession, sessionSaved])

  useEffect(() => {
    const handler = async () => {
      try {
        const backendSessions = await sessionController.list()
        const localSessions = await chatDB.loadSessions()
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
        setSessions(merged)
      } catch (err) {
        useErrorStore.getState().addError(err, { source: 'Chat Sessions' })
        const localSessions = await chatDB.loadSessions()
        setSessions(localSessions)
      }
    }
    handler()
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
      if (e.detail?.text) sendMessage(e.detail.text)
    }
    window.addEventListener('send-text', handler as EventListener)
    return () => window.removeEventListener('send-text', handler as EventListener)
  })

  // ── Computed ──────────────────────────────────────────────────────────────

  const sidebarConversations: Conversation[] = (Array.isArray(sessions) ? sessions : []).map(s => ({
    id: s.id, name: s.name || 'Untitled', session_id: s.id,
    created_at: s.createdAt || new Date().toISOString(),
    updated_at: s.updatedAt || new Date().toISOString(),
    pinned: Boolean(s.pinned), starred: Boolean(s.starred),
    message_count: Array.isArray(s.messages) ? s.messages.length : 0,
    messages: (Array.isArray(s.messages) ? s.messages : []).map(m => ({
      id: String(m.id || `msg_${Date.now()}`), role: String(m.role || 'user'), content: String(m.content || ''),
      timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : String(m.timestamp || Date.now()),
    })),
    synced: Boolean(s.synced),
  }))

  return {
    messages, setMessages,
    input, setInput,
    loading, setLoading,
    images,
    sessionSaved,
    sessions,
    currentError, setCurrentError,
    messagesRef,
    loadingRef,
    sessionIdRef,
    userIdRef,
    handleRegenerateRef,
    newChatRef,
    sendMessageRef,
    saveSessionToStorage,
    sendMessage,
    loadSession,
    newChat,
    deleteSession, starSession, pinSession, renameSession, duplicateSession,
    handleRegenerate,
    handleThumbsUp, handleThumbsDown,
    handleEditMessage,
    handleAddImage, handleRemoveImage,
    handleCopy,
    handleRetry,
    handleSuggestionClick,
    handleExportMarkdown,
    sidebarConversations,
  }
}
