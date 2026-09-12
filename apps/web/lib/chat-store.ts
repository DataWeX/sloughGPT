'use client'

/**
 * ChatStore — Zustand-based state management for the chat interface.
 *
 * Replaces the monolithic useChatMessages hook with a composable store.
 * The token accumulator and streaming pipeline operate on this store
 * directly via getState()/setState().
 */

import { create } from 'zustand'
import { startTransition } from 'react'
import { streamChatResponse, type ToolCallEvent } from '@/lib/stream-chat-response'
import {
  cleanStreamedContent, stripAssistantPrefix, getOrCreateUserId,
  generateSessionId, CURRENT_SESSION_KEY, buildLocalPrompt,
  type ChatMessage, type ImageAttachment, type ChatSession,
} from '@/lib/chat-utils'
import { logger } from '@/lib/dev-log'
import { extractErrorMessage } from '@/lib/error-utils'
import { getErrorInfo } from '@/features/chat/components/feedback/ErrorBanner'
import { chatController } from '@/lib/chat-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { multimodalController } from '@/lib/multimodal-controller'
import { chatDB } from '@/lib/db'
import { useAppStore, getKnowledgeContext } from '@/lib/store'
import { publishMemoryEvent } from '@/lib/memory-events'
import type { SoulNetWebGPU, SoulTransformerWebGPU } from '@/lib/soulnet-webgpu'
import type { AgentDef } from '@/lib/agents'
import type { Soul } from '@/lib/souls-controller'

const _log = logger.child('chat-store')

// ── Types ────────────────────────────────────────────────────────────────────

export interface ContextLayer {
  type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'
  label: string
  detail?: string
}

export interface RagVerificationInfo {
  confidence: number
  is_verified: boolean
  hallucination_rate: number
  citations: string
  grounded_claims: number
  hallucinated_claims: number
}

export type ErrorInfo = ReturnType<typeof getErrorInfo>

// ── Config (set once on init, read via getState) ─────────────────────────────

export interface ChatConfig {
  model: string
  temperature: number
  maxTokens: number
  currentSoul: Soul | null
  currentAgent: AgentDef | null
  useLocalEngine: boolean
  engineRef: React.MutableRefObject<SoulNetWebGPU | SoulTransformerWebGPU | null>
  engineLoadingRef: React.MutableRefObject<boolean>
  initLocalEngine: () => Promise<boolean>
  customSystemPrompt: string
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
  recordFeedback: (params: {
    userMessage: string
    assistantResponse: string
    rating: 'thumbs_up' | 'thumbs_down'
    conversationId?: string
    qualityScore?: number
    userId?: string
  }) => Promise<boolean>
  onVisionUpdate: (caps: unknown, history: string[], vocab: number | undefined) => void
  onKnowledgeUpdate: (ctx: { count: number; context: string }) => void
}

// ── Store State ──────────────────────────────────────────────────────────────

export interface ChatState {
  // Messages
  messages: ChatMessage[]
  loading: boolean
  sessionLoading: boolean
  streamingMessageId: string | null
  toolEvents: ToolCallEvent[]
  ragVerification: RagVerificationInfo | null
  contextLayers: ContextLayer[]
  pendingToolApproval: { toolName: string; args?: Record<string, unknown> } | null
  currentError: ErrorInfo | null

  // Input
  input: string
  images: ImageAttachment[]

  // Session
  sessionId: string
  userId: string
  sidebarConversations: ChatSession[]
  sessionSaved: boolean

  // Selection
  selectedMessageIds: Set<string>
  selectionMode: boolean

  // Config (mutable, set via init)
  _config: ChatConfig | null
}

// ── Store Actions ────────────────────────────────────────────────────────────

export interface ChatActions {
  // Core
  init: (config: ChatConfig) => void
  sendMessage: (overrideText?: string) => Promise<void>
  stop: () => void
  regenerate: (fromMessageId?: string) => Promise<void>
  regenerateWithOptions: (fromMessageId: string, options: { temperature?: number; maxTokens?: number }) => Promise<void>
  newChat: () => void
  handleRetry: () => void

  // Session
  loadSession: (id: string) => Promise<void>
  deleteSession: (id: string) => Promise<void>
  starSession: (id: string, starred: boolean) => Promise<void>
  pinSession: (id: string, pinned: boolean) => Promise<void>
  archiveSession: (id: string, archived: boolean) => Promise<void>
  renameSession: (id: string, newName: string) => Promise<void>
  duplicateSession: (id: string) => Promise<void>

  // Input
  setInput: (value: string | ((prev: string) => string)) => void
  setMessages: ( updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void
  setLoading: (loading: boolean) => void
  setCurrentError: (error: ErrorInfo | null) => void

  // Images
  handleAddImage: (dataUrl: string) => void
  handleRemoveImage: (id: string) => void

  // Feedback
  handleThumbsUp: (messageId: string) => void
  handleThumbsDown: (messageId: string) => void

  // Edit
  handleEditMessage: (messageId: string, newContent: string) => void

  // Reactions
  handleReact: (messageId: string, emoji: string) => void
  handlePin: (messageId: string) => void

  // Selection
  toggleSelectionMode: () => void
  toggleMessageSelection: (messageId: string) => void
  selectAllMessages: () => void
  clearSelection: () => void
  deleteSelectedMessages: () => void

  // UI
  handleSuggestionClick: (text: string) => void
  handleCopy: (text: string) => void
  handleToolApproval: (approved: boolean) => void
  cancelStream: () => void
  approveTool: (toolName: string, approved: boolean) => Promise<void>
  injectContext: (context: string) => Promise<void>
}

// ── Token Accumulator ────────────────────────────────────────────────────────
// Module-level: buffers tokens and flushes via requestAnimationFrame.
// Only touches the streaming message (O(1) splice, not O(n) map).

const tokenBuf: { id: string; text: string }[] = []
let rafId = 0
let pendingFlush = false

function flushTokens() {
  pendingFlush = false
  if (tokenBuf.length === 0) return
  const batch = tokenBuf.splice(0)
  const byId = new Map<string, string>()
  for (const { id, text } of batch) {
    byId.set(id, (byId.get(id) || '') + text)
  }

  const { messages } = useChatStore.getState()
  const lastIdx = messages.length - 1
  if (lastIdx < 0) return
  const last = messages[lastIdx]
  const delta = byId.get(last.id)
  if (!delta) return

  const content = last.content === 'Thinking...' ? '' : last.content
  const updated = [...messages.slice(0, lastIdx), { ...last, content: content + delta }]

  startTransition(() => {
    useChatStore.setState({ messages: updated })
  })

  // Auto-save during streaming
  const state = useChatStore.getState()
  const now = Date.now()
  if (state.sessionId && now - (_lastSaveTs || 0) > 500) {
    _lastSaveTs = now
    chatDB.saveSession({ id: state.sessionId, name: '', messages: updated, createdAt: '', updatedAt: '', synced: false, starred: false, pinned: false }).catch(() => {})
  }
}

let _lastSaveTs = 0

function scheduleFlush() {
  if (pendingFlush) return
  pendingFlush = true
  rafId = requestAnimationFrame(flushTokens)
}

function cancelFlush() {
  if (rafId) cancelAnimationFrame(rafId)
  pendingFlush = false
  if (tokenBuf.length > 0) flushTokens()
}

function resetFlush() {
  cancelFlush()
  tokenBuf.length = 0
}

// ── Store Implementation ─────────────────────────────────────────────────────

export const useChatStore = create<ChatState & ChatActions>()((set, get) => ({
  // ── Initial State ──
  messages: [],
  loading: false,
  sessionLoading: false,
  streamingMessageId: null,
  toolEvents: [],
  ragVerification: null,
  contextLayers: [],
  pendingToolApproval: null,
  currentError: null,
  input: '',
  images: [],
  sessionId: '',
  userId: 'default',
  sidebarConversations: [],
  sessionSaved: false,
  selectedMessageIds: new Set(),
  selectionMode: false,
  _config: null,

  // ── Init ──
  init: (config) => {
    set({ _config: config })

    // Initialize session ID
    chatDB.getKV<string>(CURRENT_SESSION_KEY).then(existing => {
      const sessionId = existing || generateSessionId()
      if (!existing) chatDB.setKV(CURRENT_SESSION_KEY, sessionId)
      set({ sessionId })
    })

    // Initialize user ID
    getOrCreateUserId().then(id => set({ userId: id }))

    // Load draft
    get().sessionId && chatDB.getDraft(get().sessionId).then(draft => {
      if (draft) set({ input: draft })
    })

    // Load existing session
    setTimeout(() => {
      chatDB.getKV<string>(CURRENT_SESSION_KEY).then(currentId => {
        if (currentId) get().loadSession(currentId)
      })
    }, 0)
  },

  // ── Set Input ──
  setInput: (value) => {
    const prev = get().input
    const next = typeof value === 'function' ? value(prev) : value
    set({ input: next })

    // Auto-save draft
    const state = get()
    if (state.sessionId) {
      chatDB.saveDraft(state.sessionId, next)
    }
  },

  // ── Set Messages ──
  setMessages: (updater) => {
    const prev = get().messages
    const next = typeof updater === 'function' ? updater(prev) : updater
    set({ messages: next })
  },

  // ── Set Loading ──
  setLoading: (loading) => set({ loading }),

  // ── Set Error ──
  setCurrentError: (error) => set({ currentError: error }),

  // ── Send Message ──
  sendMessage: async (overrideText) => {
    const state = get()
    const config = state._config
    if (!config) return

    const text = overrideText ?? state.input
    if ((!text.trim() && state.images.length === 0) || state.loading) return

    const userImages = [...state.images]
    const appState = useAppStore.getState()
    const customContext = appState.settings.customContext

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
      images: userImages.length > 0 ? userImages : undefined,
    }
    const assistantId = crypto.randomUUID()
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    // Batch state update
    startTransition(() => {
      set({
        messages: [...state.messages, userMessage, assistantMessage],
        input: '',
        images: [],
        currentError: null,
        toolEvents: [],
        ragVerification: null,
        contextLayers: [],
        loading: true,
        streamingMessageId: assistantId,
      })
    })

    chatDB.deleteDraft(state.sessionId)

    // Auto-name conversation from first user message
    if (state.messages.length === 0 && text.trim()) {
      const title = text.trim().slice(0, 50).replace(/[^\w\s-]/g, '').trim()
      if (title.length > 5) {
        get().renameSession(state.sessionId, title)
      }
    }

    // Build system prompt
    const parts: string[] = []
    if (config.customSystemPrompt) parts.push(`[System Override]\n${config.customSystemPrompt}`)
    if (customContext) parts.push(`[Custom Instructions]\n${customContext}`)
    if (config.currentSoul) {
      parts.push(`[Personality: ${config.currentSoul.name}]`)
      if (config.currentSoul.description) parts.push(config.currentSoul.description)
      if (config.currentSoul.traits?.length) parts.push(`Traits: ${config.currentSoul.traits.join(', ')}`)
    }
    if (config.currentAgent) {
      parts.push(`[Role: ${config.currentAgent.name}]`)
      if (config.currentAgent.description) parts.push(config.currentAgent.description)
    }
    const systemPrompt = parts.join('\n\n')
    const knowledgeFacts = appState.injectedKnowledge.map((k: { content: string }) => k.content)

    // Build context layers
    const initialContextLayers: ContextLayer[] = []
    if (config.currentSoul) initialContextLayers.push({ type: 'soul', label: `Personality: ${config.currentSoul.name}`, detail: config.currentSoul.description })
    if (config.currentAgent) initialContextLayers.push({ type: 'system', label: `Agent: ${config.currentAgent.name}`, detail: config.currentAgent.description })
    if (knowledgeFacts.length > 0) initialContextLayers.push({ type: 'knowledge', label: 'Knowledge context', detail: `${knowledgeFacts.length} facts injected` })
    set({ contextLayers: initialContextLayers })

    const messagesWithNew = [...state.messages, userMessage, assistantMessage]

    // Save session
    chatDB.saveSession({
      id: state.sessionId, name: '', messages: messagesWithNew,
      createdAt: '', updatedAt: '', synced: false, starred: false, pinned: false,
    }).catch(() => {})

    // Local engine path
    if (config.useLocalEngine && !config.engineRef.current && !config.engineLoadingRef.current) {
      await config.initLocalEngine()
    }

    const loadingController = new AbortController()

    try {
      if (config.useLocalEngine && config.engineRef.current) {
        const prompt = buildLocalPrompt(messagesWithNew, systemPrompt)
        let assistantContentLen = 0
        const signal = loadingController.signal
        for await (const token of config.engineRef.current.generate(prompt, config.maxTokens, config.temperature)) {
          if (signal.aborted) break
          let cleanedToken = token
          if (assistantContentLen < 50) {
            cleanedToken = stripAssistantPrefix(cleanedToken)
            cleanedToken = cleanStreamedContent(cleanedToken)
          }
          assistantContentLen += cleanedToken.length
          tokenBuf.push({ id: assistantId, text: cleanedToken })
          scheduleFlush()
        }
      } else {
        let assistantContentLen = 0
        let streamComplete = false

        await streamChatResponse({
          messages: messagesWithNew
            .filter(m => m.content.trim().length > 0 || m.role === 'user')
            .map(m => ({ role: m.role, content: m.content })),
          model: config.model,
          systemPrompt,
          maxTokens: config.maxTokens,
          temperature: config.temperature,
          userId: state.userId,
          sessionId: state.sessionId,
          images: userImages.length > 0 ? userImages.map(img => img.dataUrl) : undefined,
          signal: loadingController.signal,
          agentId: config.currentAgent?.id || undefined,
          knowledge: knowledgeFacts.length > 0 ? knowledgeFacts : undefined,
          onToken: (token: string) => {
            let cleanedToken = token
            if (assistantContentLen < 50) {
              cleanedToken = stripAssistantPrefix(cleanedToken)
              cleanedToken = cleanStreamedContent(cleanedToken)
            }
            assistantContentLen += cleanedToken.length
            tokenBuf.push({ id: assistantId, text: cleanedToken })
            scheduleFlush()
          },
          onComplete: () => {
            streamComplete = true
            cancelFlush()
            startTransition(() => {
              set(s => ({
                loading: false,
                sessionSaved: true,
                messages: s.messages.map(m =>
                  m.id === assistantId && m.content === 'Thinking...' ? { ...m, content: '' } : m
                ),
              }))
            })
          },
          onError: (status, text, opts) => {
            cancelFlush()
            const errorInfo = getErrorInfo(status, text || 'Stream error', opts)
            startTransition(() => {
              set(s => ({
                currentError: errorInfo,
                loading: false,
                messages: s.messages.map(msg =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
                    : msg
                ),
              }))
            })
          },
          onKnowledge: (source, count) => {
            config.showToast(`Knowledge: ${count} facts from ${source}`, 'info')
            set(s => ({
              contextLayers: [...s.contextLayers, { type: 'knowledge' as const, label: `Knowledge: ${source}`, detail: `${count} facts` }],
            }))
          },
          onMemory: (info) => {
            publishMemoryEvent(info)
            if (info.stored) {
              const list = (info.facts?.length ? info.facts : info.fact ? [info.fact] : []) as string[]
              const first = list[0]
              const extra = list.length > 1 ? ` +${list.length - 1} more` : ''
              const shown = first && first.length > 140 ? `${first.slice(0, 140)}…` : first
              config.showToast(shown ? `Remembered: ${shown}${extra}` : 'New fact saved to memory', 'success')
              set(s => ({
                contextLayers: [...s.contextLayers, { type: 'memory' as const, label: 'Memory updated', detail: first ? `${list.length} fact${list.length > 1 ? 's' : ''} stored` : undefined }],
              }))
            }
          },
          onThinking: () => {
            startTransition(() => {
              set(s => ({
                messages: s.messages.map(msg =>
                  msg.id === assistantId && !msg.content ? { ...msg, content: 'Thinking...' } : msg
                ),
              }))
            })
          },
          onToolCall: (event) => {
            set(s => ({
              toolEvents: [...s.toolEvents, event],
              contextLayers: [...s.contextLayers, { type: 'tool' as const, label: `Tool: ${event.tool}`, detail: event.status }],
            }))
            if (event.status === 'executing' && event.args) {
              const autoApprove = useAppStore.getState().settings.autoApproveTools
              if (autoApprove) {
                chatController.approveTool(state.sessionId, event.tool, true)
              } else {
                set({ pendingToolApproval: { toolName: event.tool, args: event.args } })
              }
            }
          },
          onRagVerification: (info) => {
            set(s => ({
              ragVerification: info,
              contextLayers: [...s.contextLayers, { type: 'rag' as const, label: 'RAG verification', detail: `${(info.confidence * 100).toFixed(0)}% confidence` }],
            }))
          },
          onControl: (event) => {
            if (event.action === 'cancelled') {
              startTransition(() => {
                set(s => ({
                  messages: s.messages.map(msg =>
                    msg.id === assistantId
                      ? { ...msg, content: msg.content || '(cancelled)', isError: true }
                      : msg
                  ),
                }))
              })
            } else if (event.action === 'context') {
              set(s => ({
                contextLayers: [...s.contextLayers, { type: 'system' as const, label: 'Context injected', detail: event.context }],
              }))
            }
          },
        })

        if (streamComplete) {
          chatController.saveSessionContext(state.sessionId, messagesWithNew.map(m => ({ role: m.role, content: m.content }))).catch(() => {})
          knowledgeController.context().then(res => {
            config.onKnowledgeUpdate({ count: res.count, context: res.context })
          }).catch(() => {})
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return
      const errorInfo = getErrorInfo(0, extractErrorMessage(err, 'Network error'))
      startTransition(() => {
        set(s => ({
          currentError: errorInfo,
          loading: false,
          messages: s.messages.map(msg =>
            msg.id === assistantId
              ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
              : msg
          ),
        }))
      })
    } finally {
      set({ loading: false, streamingMessageId: null })
    }
  },

  // ── Stop ──
  stop: () => {
    cancelFlush()
    const state = get()
    if (state._config) {
      chatController.cancelStream(state.sessionId).catch(() => {})
    }
    startTransition(() => {
      set({ loading: false, streamingMessageId: null, pendingToolApproval: null })
    })
  },

  // ── Regenerate ──
  regenerate: async (fromMessageId) => {
    const state = get()
    const config = state._config
    if (!config || state.messages.length < 2) return

    const targetIdx = fromMessageId
      ? state.messages.findIndex(m => m.id === fromMessageId)
      : state.messages.findLastIndex(m => m.role === 'assistant')

    if (targetIdx <= 0) return

    const truncated = state.messages.slice(0, targetIdx + 1)
    const target = state.messages[targetIdx]

    // Reset target message
    if (targetIdx < state.messages.length - 1) {
      set({ messages: [...state.messages.slice(0, targetIdx), { ...target, content: '', isError: false }] })
    } else {
      set({ messages: state.messages.map(m => m.id === target.id ? { ...m, content: '', isError: false } : m) })
    }

    set({ loading: true, currentError: null })

    const parts: string[] = []
    if (config.customSystemPrompt) parts.push(`[System Override]\n${config.customSystemPrompt}`)
    const appState = useAppStore.getState()
    const customContext = appState.settings.customContext
    if (customContext) parts.push(`[Custom Instructions]\n${customContext}`)
    if (config.currentSoul) {
      parts.push(`[Personality: ${config.currentSoul.name}]`)
      if (config.currentSoul.description) parts.push(config.currentSoul.description)
      if (config.currentSoul.traits?.length) parts.push(`Traits: ${config.currentSoul.traits.join(', ')}`)
    }
    const systemPrompt = parts.join('\n\n')
    const knowledgeFacts = appState.injectedKnowledge.map((k: { content: string }) => k.content)

    const loadingController = new AbortController()

    try {
      let assistantContentLen = 0
      await streamChatResponse({
        messages: truncated.map(m => ({ role: m.role, content: m.content })),
        model: config.model,
        systemPrompt,
        maxTokens: config.maxTokens,
        temperature: config.temperature,
        userId: state.userId,
        sessionId: state.sessionId,
        signal: loadingController.signal,
        onToken: (token) => {
          let cleanedToken = token
          if (assistantContentLen < 50) {
            cleanedToken = stripAssistantPrefix(cleanedToken)
            cleanedToken = cleanStreamedContent(cleanedToken)
          }
          assistantContentLen += cleanedToken.length
          tokenBuf.push({ id: target.id, text: cleanedToken })
          scheduleFlush()
        },
        onComplete: () => {
          cancelFlush()
          set(s => ({
            loading: false,
            messages: s.messages.map(m =>
              m.id === target.id && !m.content ? { ...m, content: '(empty response)' } : m
            ),
          }))
        },
        onError: (status, text, opts) => {
          cancelFlush()
          const errorInfo = getErrorInfo(status, text || 'Stream error', opts)
          set(s => ({
            currentError: errorInfo,
            loading: false,
            messages: s.messages.map(msg =>
              msg.id === target.id
                ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
                : msg
            ),
          }))
        },
      })
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return
      set(s => ({
        currentError: getErrorInfo(0, extractErrorMessage(err, 'Network error')),
        loading: false,
        messages: s.messages.map(msg =>
          msg.id === target.id
            ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
            : msg
        ),
      }))
    } finally {
      set({ loading: false, streamingMessageId: null })
    }
  },

  // ── Regenerate with Options ──
  regenerateWithOptions: async (fromMessageId, options) => {
    const state = get()
    const config = state._config
    if (!config || state.messages.length < 2) return

    const targetIdx = fromMessageId
      ? state.messages.findIndex(m => m.id === fromMessageId)
      : state.messages.findLastIndex(m => m.role === 'assistant')

    if (targetIdx <= 0) return

    const truncated = state.messages.slice(0, targetIdx + 1)
    const target = state.messages[targetIdx]

    // Reset target message
    if (targetIdx < state.messages.length - 1) {
      set({ messages: [...state.messages.slice(0, targetIdx), { ...target, content: '', isError: false }] })
    } else {
      set({ messages: state.messages.map(m => m.id === target.id ? { ...m, content: '', isError: false } : m) })
    }

    set({ loading: true, currentError: null })

    const parts: string[] = []
    if (config.customSystemPrompt) parts.push(`[System Override]\n${config.customSystemPrompt}`)
    const appState = useAppStore.getState()
    const customContext = appState.settings.customContext
    if (customContext) parts.push(`[Custom Instructions]\n${customContext}`)
    if (config.currentSoul) {
      parts.push(`[Personality: ${config.currentSoul.name}]`)
      if (config.currentSoul.description) parts.push(config.currentSoul.description)
      if (config.currentSoul.traits?.length) parts.push(`Traits: ${config.currentSoul.traits.join(', ')}`)
    }
    const systemPrompt = parts.join('\n\n')

    const loadingController = new AbortController()

    try {
      let assistantContentLen = 0
      await streamChatResponse({
        messages: truncated.map(m => ({ role: m.role, content: m.content })),
        model: config.model,
        systemPrompt,
        maxTokens: options?.maxTokens ?? config.maxTokens,
        temperature: options?.temperature ?? config.temperature,
        userId: state.userId,
        sessionId: state.sessionId,
        signal: loadingController.signal,
        onToken: (token) => {
          let cleanedToken = token
          if (assistantContentLen < 50) {
            cleanedToken = stripAssistantPrefix(cleanedToken)
            cleanedToken = cleanStreamedContent(cleanedToken)
          }
          assistantContentLen += cleanedToken.length
          tokenBuf.push({ id: target.id, text: cleanedToken })
          scheduleFlush()
        },
        onComplete: () => {
          cancelFlush()
          set(s => ({
            loading: false,
            messages: s.messages.map(m =>
              m.id === target.id && !m.content ? { ...m, content: '(empty response)' } : m
            ),
          }))
        },
        onError: (status, text, opts) => {
          cancelFlush()
          const errorInfo = getErrorInfo(status, text || 'Stream error', opts)
          set(s => ({
            currentError: errorInfo,
            loading: false,
            messages: s.messages.map(msg =>
              msg.id === target.id
                ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
                : msg
            ),
          }))
        },
      })
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return
      set(s => ({
        currentError: getErrorInfo(0, extractErrorMessage(err, 'Network error')),
        loading: false,
        messages: s.messages.map(msg =>
          msg.id === target.id
            ? { ...msg, content: msg.content || '(response interrupted)', isError: true }
            : msg
        ),
      }))
    } finally {
      set({ loading: false, streamingMessageId: null })
    }
  },

  // ── New Chat ──
  newChat: () => {
    const newId = generateSessionId()
    chatDB.setKV(CURRENT_SESSION_KEY, newId)
    chatDB.deleteDraft(newId)
    resetFlush()
    startTransition(() => {
      set({
        messages: [],
        input: '',
        sessionSaved: false,
        sessionId: newId,
        currentError: null,
        toolEvents: [],
        ragVerification: null,
        contextLayers: [],
        pendingToolApproval: null,
        selectedMessageIds: new Set(),
        selectionMode: false,
      })
    })
    get()._config?.showToast('New chat started')
  },

  // ── Handle Retry ──
  handleRetry: () => {
    const state = get()
    set({ currentError: null })
    const lastUser = state.messages.findLast(m => m.role === 'user')
    if (lastUser?.content) {
      get().sendMessage(lastUser.content)
    } else if (state.input.trim()) {
      get().sendMessage()
    }
  },

  // ── Session Operations ──
  loadSession: async (id) => {
    set({ sessionLoading: true })
    try {
      const sessions = await chatDB.loadSessions()
      const session = sessions.find(s => s.id === id)
      if (session) {
        startTransition(() => {
          set({
            messages: session.messages || [],
            sessionId: id,
            sessionSaved: true,
          })
        })
        chatDB.setKV(CURRENT_SESSION_KEY, id)
      }
    } catch (err) {
      _log.error('Failed to load session', { error: String(err) })
    } finally {
      set({ sessionLoading: false })
    }
  },

  deleteSession: async (id) => {
    try {
      await chatDB.deleteSession(id)
      set(s => ({
        sidebarConversations: s.sidebarConversations.filter(c => c.id !== id),
      }))
      if (get().sessionId === id) get().newChat()
    } catch (err) {
      _log.error('Failed to delete session', { error: String(err) })
    }
  },

  starSession: async (id, starred) => {
    set(s => ({
      sidebarConversations: s.sidebarConversations.map(c => c.id === id ? { ...c, starred } : c),
    }))
  },

  pinSession: async (id, pinned) => {
    set(s => ({
      sidebarConversations: s.sidebarConversations.map(c => c.id === id ? { ...c, pinned } : c),
    }))
  },

  archiveSession: async (id, archived) => {
    set(s => ({
      sidebarConversations: s.sidebarConversations.map(c => c.id === id ? { ...c, archived } : c),
    }))
  },

  renameSession: async (id, newName) => {
    set(s => ({
      sidebarConversations: s.sidebarConversations.map(c => c.id === id ? { ...c, name: newName } : c),
    }))
  },

  duplicateSession: async (id) => {
    const state = get()
    const session = state.sidebarConversations.find(c => c.id === id)
    if (!session) return
    const newId = generateSessionId()
    const duplicate: ChatSession = {
      ...session,
      id: newId,
      name: `${session.name} (copy)`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }
    await chatDB.saveSession(duplicate)
    set(s => ({
      sidebarConversations: [...s.sidebarConversations, duplicate],
    }))
  },

  // ── Images ──
  handleAddImage: (dataUrl) => {
    const newImage: ImageAttachment = {
      id: crypto.randomUUID(),
      dataUrl,
      name: `image-${Date.now()}.png`,
    }
    set(s => ({ images: [...s.images, newImage] }))

    const config = get()._config
    if (config) {
      multimodalController.trainImage(dataUrl, newImage.name).then(res => {
        multimodalController.getCapabilities().then(caps => {
          multimodalController.getTrainingReport().then(r => {
            config.onVisionUpdate(caps, r.caption_history || [], r.vocab_size)
          }).catch(() => {})
        }).catch(() => {})
      }).catch(() => {})
    }
  },

  handleRemoveImage: (id) => {
    set(s => ({ images: s.images.filter(img => img.id !== id) }))
  },

  // ── Feedback ──
  handleThumbsUp: (messageId) => {
    const state = get()
    const config = state._config
    if (!config) return
    const msgIdx = state.messages.findIndex(m => m.id === messageId)
    const userMsg = msgIdx > 0 ? state.messages[msgIdx - 1] : null
    const assistantMsg = state.messages[msgIdx]
    config.recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_up',
      conversationId: state.sessionId,
      userId: state.userId,
    }).then(ok => {
      config.showToast(ok ? 'Thanks for the feedback!' : 'Could not submit feedback', ok ? 'success' : 'error')
    })
  },

  handleThumbsDown: (messageId) => {
    const state = get()
    const config = state._config
    if (!config) return
    const msgIdx = state.messages.findIndex(m => m.id === messageId)
    const userMsg = msgIdx > 0 ? state.messages[msgIdx - 1] : null
    const assistantMsg = state.messages[msgIdx]
    config.recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_down',
      conversationId: state.sessionId,
      userId: state.userId,
    }).then(ok => {
      config.showToast(ok ? 'Thanks for the feedback!' : 'Could not submit feedback', ok ? 'success' : 'error')
    })
  },

  // ── Edit ──
  handleEditMessage: (messageId, newContent) => {
    const state = get()
    const msgIndex = state.messages.findIndex(m => m.id === messageId)
    if (msgIndex === -1) return
    set({ messages: state.messages.slice(0, msgIndex), loading: false, currentError: null })
    setTimeout(() => get().sendMessage(newContent), 0)
  },

  // ── Reactions ──
  handleReact: (messageId, emoji) => {
    set(s => ({
      messages: s.messages.map(msg => {
        if (msg.id !== messageId) return msg
        const reactions = { ...msg.reactions }
        reactions[emoji] = (reactions[emoji] || 0) + 1
        return { ...msg, reactions }
      }),
    }))
  },

  handlePin: (messageId) => {
    set(s => ({
      messages: s.messages.map(msg =>
        msg.id === messageId ? { ...msg, pinned: !msg.pinned } : msg
      ),
    }))
  },

  // ── Selection ──
  toggleSelectionMode: () => {
    set(s => ({
      selectionMode: !s.selectionMode,
      selectedMessageIds: new Set(),
    }))
  },

  toggleMessageSelection: (messageId) => {
    set(s => {
      const next = new Set(s.selectedMessageIds)
      if (next.has(messageId)) next.delete(messageId)
      else next.add(messageId)
      return { selectedMessageIds: next }
    })
  },

  selectAllMessages: () => {
    set(s => ({
      selectedMessageIds: new Set(s.messages.filter(m => !m.isError).map(m => m.id)),
    }))
  },

  clearSelection: () => {
    set({ selectedMessageIds: new Set() })
  },

  deleteSelectedMessages: () => {
    set(s => ({
      messages: s.messages.filter(msg => !s.selectedMessageIds.has(msg.id)),
      selectedMessageIds: new Set(),
      selectionMode: false,
    }))
  },

  // ── UI ──
  handleSuggestionClick: (text) => {
    set({ input: text })
    setTimeout(() => get().sendMessage(text), 0)
  },

  handleCopy: (text) => {
    get()._config?.showToast('Copied to clipboard')
  },

  handleToolApproval: (approved) => {
    const state = get()
    if (state.pendingToolApproval) {
      chatController.approveTool(state.sessionId, state.pendingToolApproval.toolName, approved)
      set({ pendingToolApproval: null })
    }
  },

  cancelStream: () => {
    chatController.cancelStream(get().sessionId).catch(() => {})
  },

  approveTool: async (toolName, approved) => {
    await chatController.approveTool(get().sessionId, toolName, approved)
  },

  injectContext: async (context) => {
    await chatController.injectContext(get().sessionId, context)
  },
}))
