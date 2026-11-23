'use client'

import { useRef, useEffect, useMemo } from 'react'
import {
  generateSessionId,
  CURRENT_SESSION_KEY,
  type ChatMessage,
  type ChatSession,
} from '@/lib/chat-utils'
import { chatDB } from '@/lib/db'
import { useErrorStore } from '@/lib/error-store'
import { useConversationScope } from '@/lib/conversation-scope-store'
import { useChatSessions } from './useChatSessions'
import { useChatStore, streamAbortRef } from '@/lib/chat-store'

interface ChatMessagesConfig {
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
}

/**
 * useChatMessages — the chat facade: a PROJECTION of ChatStore.
 *
 * Stage 3 (send-path unification): every action this facade exposes IS the
 * store's action — one sendMessage, one regenerate pipeline, one token
 * accumulator, one AbortController (`streamAbortRef`). The controller-era
 * duplicate implementations (send, regenerate, flush, retry/edit/suggestion,
 * thumbs, images, copy/export, selection) are gone; they lived here and in
 * chat-store.ts at the same time, and `stop` cancelled the store's accumulator
 * while the live stream buffered into this hook's.
 *
 * What remains hook-owned: session list operations (useChatSessions — it owns
 * the list state and loadSession's guarded transcript swap), mount-time
 * bootstrap, and the ref handles keyboard shortcuts and window events read.
 */
export function useChatMessages(config: ChatMessagesConfig) {
  const { showToast } = config

  // ── Core state — ChatStore is the single authority (card 20261116_124020) ──
  // These used to be local useState copies mirrored one-way into the store by
  // useChatStoreBridge — readers on the store and writers on this hook could
  // desync (stale panes, scrambled load order). The hook now reads/writes the
  // store directly: one state, both APIs, no mirror.
  const messages = useChatStore((s) => s.messages)
  const input = useChatStore((s) => s.input)
  const loading = useChatStore((s) => s.loading)
  const sessionLoading = useChatStore((s) => s.sessionLoading)
  const images = useChatStore((s) => s.images)
  const sessionSaved = useChatStore((s) => s.sessionSaved)
  const currentError = useChatStore((s) => s.currentError)
  const contextLayers = useChatStore((s) => s.contextLayers)
  const toolEvents = useChatStore((s) => s.toolEvents)
  const ragVerification = useChatStore((s) => s.ragVerification)
  const pendingToolApproval = useChatStore((s) => s.pendingToolApproval)
  const selectedMessageIds = useChatStore((s) => s.selectedMessageIds)
  const selectionMode = useChatStore((s) => s.selectionMode)

  const setMessages = useChatStore((s) => s.setMessages)
  const setInput = useChatStore((s) => s.setInput)
  const setLoading = useChatStore((s) => s.setLoading)
  const setCurrentError = useChatStore((s) => s.setCurrentError)
  const setSessionSaved = useChatStore((s) => s.setSessionSaved)
  const setSessionLoading = useChatStore((s) => s.setSessionLoading)

  // ── Refs for callback access ──
  const messagesRef = useRef<ChatMessage[]>([])
  /** The live stream's abort handle — the store's own (see streamAbortRef):
   *  store sendMessage/regenerate assign it, stop()/Escape/the stop button
   *  abort it. A hook-local controller here was the stage-2 split-brain. */
  const loadingRef = streamAbortRef
  /** sessionId is store state. A plain ref would go stale the moment the
   *  store's newChat/loadSession moved the claim — reads must be live. */
  const sessionIdRef = useMemo<React.MutableRefObject<string>>(
    () => ({
      get current() {
        return useChatStore.getState().sessionId
      },
      set current(value: string) {
        useChatStore.setState({ sessionId: value })
      },
    }),
    [],
  )
  const userIdRef = useRef<string>('default')
  const sendMessageRef = useRef<(overrideText?: string) => Promise<void>>(async () => {})
  const handleRegenerateRef = useRef<(() => Promise<void>) | null>(null)
  const newChatRef = useRef<(() => void) | null>(null)
  const draftTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Session operations (delegated, hook-owned list) ──
  const sessions = useChatSessions({
    setMessages,
    setInput,
    setSessionSaved,
    setSessionLoading,
    sessionIdRef,
    messagesRef,
    showToast,
  })

  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions

  // ── Actions — every one of these IS the store action (stage 3) ──
  const sendMessage = useChatStore((s) => s.sendMessage)
  const newChat = useChatStore((s) => s.newChat)
  const handleRegenerate = useChatStore((s) => s.regenerate)
  const handleRegenerateWithOptions = useChatStore((s) => s.regenerateWithOptions)
  const handleRetry = useChatStore((s) => s.handleRetry)
  const handleEditMessage = useChatStore((s) => s.handleEditMessage)
  const handleSuggestionClick = useChatStore((s) => s.handleSuggestionClick)
  const handleThumbsUp = useChatStore((s) => s.handleThumbsUp)
  const handleThumbsDown = useChatStore((s) => s.handleThumbsDown)
  const handleAddImage = useChatStore((s) => s.handleAddImage)
  const handleRemoveImage = useChatStore((s) => s.handleRemoveImage)
  const handleCopy = useChatStore((s) => s.handleCopy)
  const handleExportMarkdown = useChatStore((s) => s.handleExportMarkdown)
  const handleCopyMarkdown = useChatStore((s) => s.handleCopyMarkdown)
  const handleReact = useChatStore((s) => s.handleReact)
  const handlePin = useChatStore((s) => s.handlePin)
  const toggleSelectionMode = useChatStore((s) => s.toggleSelectionMode)
  const toggleMessageSelection = useChatStore((s) => s.toggleMessageSelection)
  const selectAllMessages = useChatStore((s) => s.selectAllMessages)
  const clearSelection = useChatStore((s) => s.clearSelection)
  const deleteSelectedMessages = useChatStore((s) => s.deleteSelectedMessages)
  const cancelStream = useChatStore((s) => s.cancelStream)
  const approveTool = useChatStore((s) => s.approveTool)
  const injectContext = useChatStore((s) => s.injectContext)
  const handleToolApproval = useChatStore((s) => s.handleToolApproval)

  // Ref handles for keyboard shortcuts and window events (stable store
  // actions — assigned once, never stale).
  sendMessageRef.current = sendMessage
  handleRegenerateRef.current = handleRegenerate
  newChatRef.current = newChat

  // Keep messagesRef in sync via effect (single source of truth is
  // store.messages — event-time readers tolerate the one-frame lag).
  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    chatDB.getKV<string>(CURRENT_SESSION_KEY).then((existing) => {
      if (existing) {
        sessionIdRef.current = existing
      } else {
        const fresh = generateSessionId()
        sessionIdRef.current = fresh
        chatDB.setKV(CURRENT_SESSION_KEY, fresh)
      }
    })
    chatDB.getDraft(sessionIdRef.current).then((savedDraft) => {
      if (savedDraft) setInput(savedDraft)
    })
  }, [])

  useEffect(() => {
    chatDB.getKV<string>(CURRENT_SESSION_KEY).then((currentId) => {
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
      return () => {
        if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
      }
    }
  }, [sessionSaved])

  useEffect(() => {
    let ignore = false
    const handler = async () => {
      try {
        const backendSessions = await (
          await import('@/lib/session-controller')
        ).sessionController.list()
        // The session index carries each conversation's AppScope delta — seed
        // the scope cache from the fetch we already made (no extra request).
        useConversationScope.getState().seed(backendSessions)
        if (ignore) return
        const localSessions = await chatDB.loadSessions()
        if (ignore) return
        const merged: ChatSession[] = [
          ...(Array.isArray(backendSessions) ? backendSessions : []).map((s): ChatSession => ({
            id: s.id,
            name: s.name || `Chat ${s.id}`,
            messages: (s.messages || []).map((m) => ({
              id: m.id || `msg_${Date.now()}`,
              role: (m.role || 'user') as 'user' | 'assistant',
              content: m.content || '',
              timestamp: new Date(m.timestamp || Date.now()),
            })),
            createdAt: s.created_at,
            updatedAt: s.updated_at,
            synced: true,
            starred: s.starred ?? false,
            pinned: s.pinned ?? false,
          })),
          ...localSessions.filter(
            (l) =>
              !Array.isArray(backendSessions) ||
              !backendSessions.find((b: { id: string }) => b.id === l.id),
          ),
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
    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    const handler = (e: CustomEvent) => {
      const { dataUrl, prompt } = e.detail
      setMessages((prev) => [
        ...prev,
        {
          id: `img-${Date.now()}`,
          role: 'user',
          content: `[Generate image: ${prompt}]`,
          timestamp: new Date(),
          images: [{ id: `gen-${Date.now()}`, dataUrl, name: 'generated.png' }],
        },
      ])
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
    messages,
    setMessages,
    input,
    setInput,
    loading,
    setLoading,
    sessionLoading,
    images,
    sessionSaved,
    currentError,
    setCurrentError,
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
    // The store calls back through this seam for send/stream auto-save — the
    // list itself stays hook-owned (see useChatStoreBridge).
    saveSessionToStorage: sessions.saveSessionToStorage,
    handleRegenerate,
    handleRegenerateWithOptions,
    handleThumbsUp,
    handleThumbsDown,
    handleEditMessage,
    handleAddImage,
    handleRemoveImage,
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
    cancelStream,
    approveTool,
    injectContext,
    pendingToolApproval,
    handleToolApproval,
  }
}
