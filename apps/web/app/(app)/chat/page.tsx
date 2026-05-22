'use client'

import { useEffect, useState, useRef, useCallback, useMemo } from 'react'
import { useApiHealth } from '@/hooks/useApiHealth'
import { API_CHAT_ENDPOINT, PUBLIC_API_URL } from '@/lib/config'
import type { Conversation } from '@/lib/session-controller'
import { agentsController } from '@/lib/agents-controller'
import { AGENTS, type AgentDef } from '@/lib/agents'
import { modelController } from '@/lib/model-controller'
import { startDownload, getDownloadStatus } from '@/lib/download-controller'
import { chatController } from '@/lib/chat-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { sessionController } from '@/lib/session-controller'
import { soulsController, type Soul, type Checkpoint } from '@/lib/souls-controller'
import { multimodalController, type MultimodalCapabilities } from '@/lib/multimodal-controller'
import { chatDB, type ChatSession as DBChatSession } from '@/lib/db'
import { useFeedbackStore } from '@/lib/feedback-store'
import { useErrorStore } from '@/lib/error-store'
import { useToastStore } from '@/lib/toast-store'
import { devDebug } from '@/lib/dev-log'
import { isMeteredConnection, connectionLabel, getNetworkInfo } from '@/lib/network'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { IconSearch, IconX, IconChevronDown, IconCheck, IconSettings, IconHeart, IconRefresh, IconAlert, IconMore, IconModel } from '@/components/ui'
import { Cpu, Server } from 'lucide-react'
import {
  ChatSettings,
  ChatArea,
  ChatSidebar,
  ErrorBanner,
  ConversationViewer,
  getErrorInfo,
  type ChatMessage,
  type ImageAttachment,
} from '@/components/chat'
import { ChatToolPanel } from '@/components/chat/ChatToolPanel'
import { VoiceChatMode } from '@/components/chat/VoiceChatMode'
import { ConversationSearch } from '@/components/chat/ConversationSearch'
import { SoulNetWebGPU, SoulTransformerWebGPU, inferArch } from '@/lib/soulnet-webgpu'

const CURRENT_SESSION_KEY = 'sloughgpt_current_conversation'
const STORAGE_KEY = 'sloughgpt_chat_conversations'
const USER_ID_KEY = 'sloughgpt_user_id'
const SESSION_CONTEXT_ENDPOINT = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/session`

function cleanStreamedContent(text: string): string {
  if (!text) return text
  
  // Remove > artifacts at line start (common in chat responses)
  let cleaned = text.replace(/^\s*>\s*/gm, '')
  
  // Remove duplicate Assistant: prefixes
  cleaned = cleaned.replace(/^(Assistant:\s*)+/i, '')
  
  // Remove leading whitespace artifacts
  cleaned = cleaned.replace(/^\s+/, '')
  
  return cleaned
}

function stripAssistantPrefix(text: string): string {
  if (!text) return text
  const prefixes = [
    /^Assistant:\s*/i,
    /^\n?Assistant:\s*/i,
    /^\s*Assistant:\s*/i,
    /^\s*>\s*Assistant:\s*/i,
  ]
  for (const prefix of prefixes) {
    if (prefix.test(text)) {
      text = text.replace(prefix, '')
    }
  }
  return text
}

function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return 'default'
  let userId = localStorage.getItem(USER_ID_KEY)
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem(USER_ID_KEY, userId)
  }
  return userId
}

interface ChatSession {
  id: string
  name: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  synced: boolean
  starred: boolean
  pinned: boolean
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false) // Always reset to false on mount
  const [showSettings, setShowSettings] = useState(false)
  const [showConversationViewer, setShowConversationViewer] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showConversationSearch, setShowConversationSearch] = useState(false)
  const [matchIndex, setMatchIndex] = useState(0)
  const chatScreenRef = useRef<HTMLDivElement>(null)

  // Compute matched message IDs for search navigation
  const matchIds = useMemo(() => {
    if (!searchQuery) return []
    const q = searchQuery.toLowerCase()
    return messages.filter(m => m.content.toLowerCase().includes(q)).map(m => m.id)
  }, [messages, searchQuery])
  const matchCount = matchIds.length
  const [model, setModel] = useState('')
  const [souls, setSouls] = useState<Soul[]>([])
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelInfoMap, setModelInfoMap] = useState<Record<string, { cached?: boolean; size_gb?: number }>>({})
  const [downloadProgress, setDownloadProgress] = useState<Record<string, { percentage: number; status: string }>>({})
  const [currentError, setCurrentError] = useState<ReturnType<typeof getErrorInfo> | null>(null)
  const [images, setImages] = useState<ImageAttachment[]>([])
  const [sessionSaved, setSessionSaved] = useState(false)
  const [currentSoul, setCurrentSoul] = useState<Soul | null>(null)
  const [streamingStats, setStreamingStats] = useState<{
    tokens: number
    timeElapsed: number
    tokensPerSecond: number
  } | undefined>(undefined)
  const [useLocalEngine, setUseLocalEngine] = useState(false)
  const [localEngineLoading, setLocalEngineLoading] = useState(false)
  const [localArchInfo, setLocalArchInfo] = useState<string | null>(null)
  const [localModelUrl, setLocalModelUrl] = useState('')
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [learnerInfo, setLearnerInfo] = useState<{
    total_tokens_ingested: number
    train_steps_completed: number
    current_loss: number
    loss_history?: Array<{ step: number; loss: number; tokens: number; timestamp: number }>
    n_embed?: number
    n_layer?: number
    n_head?: number
    arch?: string
  } | null>(null)
  const [learnerTraining, setLearnerTraining] = useState(false)
  const [visionCaps, setVisionCaps] = useState<MultimodalCapabilities | null>(null)
  const [visionCaptionHistory, setVisionCaptionHistory] = useState<string[]>([])
  const [visionVocabSize, setVisionVocabSize] = useState<number | undefined>(undefined)
  const [agents, setAgents] = useState<AgentDef[]>([])
  const [currentAgent, setCurrentAgent] = useState<{ id: string; name: string; description?: string; instructions: string } | null>(null)

  const [checkpoints, setCheckpoints] = useState<Array<{ name: string; loss?: number; traits?: string[]; is_loaded?: boolean; eval_verdict?: string }>>([])
  const [currentCheckpoint, setCurrentCheckpoint] = useState<string | undefined>(undefined)
  const [toolPanelOpen, setToolPanelOpen] = useState(true)
  const [voiceMode, setVoiceMode] = useState(false)
  const { state: health, refresh: refreshHealth } = useApiHealth()
  const { recordFeedback, fetchStats, fetchAdapterStats } = useFeedbackStore()
  const engineRef = useRef<SoulNetWebGPU | SoulTransformerWebGPU | null>(null)
  const engineLoadingRef = useRef(false)
  const sessionIdRef = useRef<string>('')
  const userIdRef = useRef<string>('default')
  const lastSaveRef = useRef<number>(0)
  const sessionCreatedRef = useRef(false)
  const messagesRef = useRef<ChatMessage[]>([])
  const loadingRef = useRef<AbortController | null>(null)
  const skipMeteredWarningRef = useRef(false)
  const streamStartRef = useRef<number>(0)
  const handleRegenerateRef = useRef<() => Promise<void>>(null as unknown as () => Promise<void>)
  const newChatRef = useRef<() => void>(null as unknown as () => void)

  // Cmd+K / Ctrl+K to toggle tools panel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setToolPanelOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Listen for global new-chat shortcut
  useEffect(() => {
    const newChatHandler = () => newChatRef.current?.()
    window.addEventListener('new-chat', newChatHandler)
    return () => window.removeEventListener('new-chat', newChatHandler)
  }, [])

  // Listen for global search-conversations shortcut
  useEffect(() => {
    const handler = () => setShowConversationSearch(true)
    window.addEventListener('search-conversations', handler)
    return () => window.removeEventListener('search-conversations', handler)
  }, [])

  // IndexedDB sessions
  const [sessions, setSessions] = useState<ChatSession[]>([])

  // Fetch multimodal capabilities and training report
  useEffect(() => {
    multimodalController.getCapabilities().then(setVisionCaps).catch(() => {})
    multimodalController.getTrainingReport().then(r => {
      setVisionCaptionHistory(r.caption_history || [])
      setVisionVocabSize(r.vocab_size)
    }).catch(() => {})
  }, [])

  // Fetch sessions from backend API and local IndexedDB, merge
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const backendSessions = await sessionController.list()
        const localSessions = await chatDB.loadSessions()
        const merged: ChatSession[] = [
          ...(Array.isArray(backendSessions) ? backendSessions : []).map((s): ChatSession => ({
            id: s.id,
            name: s.name || `Chat ${s.id}`,
            messages: (s.messages || []).map(m => ({
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
          ...localSessions.filter(l => !Array.isArray(backendSessions) || !backendSessions.find((b: { id: string }) => b.id === l.id)),
        ]
        setSessions(merged)
      } catch (err) {
        useErrorStore.getState().addError(err, { source: 'Chat Sessions' })
        const localSessions = await chatDB.loadSessions()
        setSessions(localSessions)
      }
    }
    loadSessions()
  }, [])

  // Generate hash ID for session
  const generateSessionId = () => {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    let hash = ''
    for (let i = 0; i < 8; i++) {
      hash += chars[Math.floor(Math.random() * chars.length)]
    }
    return `chat_${hash}`
  }

  useEffect(() => {
    sessionIdRef.current = localStorage.getItem(CURRENT_SESSION_KEY) || generateSessionId()
    userIdRef.current = getOrCreateUserId()
    // Fetch initial feedback stats
    fetchStats()
    fetchAdapterStats()
    // Sync model from health if loaded
    if (health && health !== 'offline' && health.model_loaded && health.model_type) {
      setModel(health.model_type)
    }
    // Fetch available models
    modelController.list().then((models) => {
      setAvailableModels(models.map(m => m.id))
      const infoMap: Record<string, { cached?: boolean; size_gb?: number }> = {}
      models.forEach(m => {
        infoMap[m.id] = { cached: m.cached, size_gb: m.size_gb }
      })
      setModelInfoMap(infoMap)
    }).catch(() => {})
    // Fetch generation config from server
    generationConfigController.get().then((config) => {
      setTemperature(config.temperature)
      setMaxTokens(config.max_new_tokens)
    }).catch(() => {})
    // Fetch souls list
    soulsController.list().then((data) => {
      setSouls(data.souls || [])
      if (data.current_soul) {
        const found = (data.souls || []).find(s => s.name === data.current_soul)
        if (found) setCurrentSoul(found)
      }
    }).catch(() => {})
    // Fetch agents list (merge local AGENTS with any backend agents)
    agentsController.list().then((data: AgentDef[]) => {
      const localAgents = Object.values(AGENTS)
      // Use backend agents if available, otherwise fall back to local
      const merged = data && data.length > 0 ? data : localAgents
      setAgents(merged)
      const savedAgentId = localStorage.getItem('sloughgpt_current_agent') || 'general'
      const found = merged.find(a => a.id === savedAgentId)
      if (found) {
        setCurrentAgent({
          id: found.id,
          name: found.name,
          description: found.description || '',
          instructions: found.instructions || '',
        })
      }
    }).catch(() => {
      // Fallback to local agents
      const localAgents = Object.values(AGENTS)
      setAgents(localAgents)
      const savedAgentId = localStorage.getItem('sloughgpt_current_agent') || 'general'
      const found = localAgents.find(a => a.id === savedAgentId)
      if (found) {
        setCurrentAgent({
          id: found.id,
          name: found.name,
          description: found.description || '',
          instructions: found.instructions || '',
        })
      }
    })
    // Fetch available .sou checkpoints for local (WebGPU) inference
    soulsController.listCheckpoints().then(({ checkpoints: ckpts }) => {
      setCheckpoints((ckpts || []).map((c: Checkpoint) => ({
        name: c.name || 'unknown',
        loss: c.loss,
        traits: c.traits ? Object.keys(c.traits) : undefined,
        is_loaded: (c as any).is_loaded || false,
        eval_verdict: c.verdict,
      })))
    }).catch(() => {})
    // Fetch learner status
    fetch(`${PUBLIC_API_URL}/learn/status`).then(r => r.json()).then(data => {
      if (data && data.total_tokens_ingested !== undefined) setLearnerInfo(data)
    }).catch(() => {})
  }, [fetchStats, fetchAdapterStats])

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const store = useToastStore.getState()
    const exists = store.toasts.some(t => t.message === message && t.type === type)
    if (!exists) {
      store.addToast(message, type)
    }
  }, [])

  // Storage: max messages to keep in IndexedDB (sliding window)
const MAX_STORAGE_MESSAGES = 40

// Save latest messages for crash recovery
  const saveSessionToStorage = async (msgs: ChatMessage[], sessionId: string) => {
    const sessionName = msgs[0]?.content?.slice(0, 30) || 'New Chat'
    
    // Keep only most recent messages for storage
    const storedMsgs = msgs.length > MAX_STORAGE_MESSAGES
      ? msgs.slice(-MAX_STORAGE_MESSAGES)
      : msgs
    
    const session: DBChatSession = {
      id: sessionId,
      name: sessionName,
      messages: storedMsgs,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      synced: false,
      starred: false,
      pinned: false,
    }
    await chatDB.saveSession(session)
    if (!sessionCreatedRef.current) {
      sessionController.create(sessionName, sessionId).catch(console.error)
      sessionCreatedRef.current = true
    } else {
      sessionController.update(sessionId, { name: sessionName }).catch(() => {})
    }
    const newSessions = await chatDB.loadSessions()
    setSessions(newSessions)
  }

  const saveSession = useCallback(async () => {
    if (messages.length === 0) return
    const sessionId = localStorage.getItem(CURRENT_SESSION_KEY) || generateSessionId()
    await saveSessionToStorage(messages, sessionId)
  }, [messages])

  // Auto-init vector store on mount (RAG auto-enables via ContextCore)
  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    fetch(`${API_URL}/vector/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: 'chromadb', dimension: 384 }),
    }).catch(() => {})
  }, [])

    const loadSession = useCallback(async (sessionId: string) => {
      const session = await chatDB.loadSession(sessionId)
      if (session) {
        const filteredMessages = session.messages.filter((msg) => {
          if (msg.role === 'assistant' && !msg.content.trim()) {
            return false
          }
          return true
        })
        // Fetch any stored messages from the backend and merge
        try {
          const remoteMsgs = await sessionController.fetchMessages(sessionId)
          // Convert to ChatMessage shape used locally
          const remoteChatMsgs = remoteMsgs.map(m => ({
            id: `remote_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: new Date(),
          }))
          // Merge remote + local, deduplicating by content+role
          const seen = new Set(remoteChatMsgs.map(m => `${m.role}:${m.content}`))
          const uniqueLocal = filteredMessages.filter(m => !seen.has(`${m.role}:${m.content}`))
          setMessages([...remoteChatMsgs, ...uniqueLocal])
        } catch {
            // Fallback to just local DB messages if remote fetch fails
            setMessages(filteredMessages)
        }
        localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
        const isComplete = filteredMessages.length > 0 && 
                          filteredMessages[filteredMessages.length - 1].role === 'assistant'
        setSessionSaved(isComplete)
        if (filteredMessages.length > 0) {
          showToast(`Loaded: ${session.name}`)
        }
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

  const initLocalEngine = useCallback(async (): Promise<boolean> => {
    if (engineRef.current || engineLoadingRef.current) return true
    if (!navigator.gpu) {
      showToast('WebGPU not available in this browser', 'error')
      devDebug('WebGPU unavailable', { navigator_gpu: false })
      return false
    }
    engineLoadingRef.current = true
    setLocalEngineLoading(true)
    try {
      if (!localModelUrl) throw new Error('No .soul file URL configured')
      const url = localModelUrl.startsWith('/auto-train/') || localModelUrl.startsWith('/sou/')
        ? `${PUBLIC_API_URL}${localModelUrl}`
        : localModelUrl
      devDebug('Fetching model for local engine', { url })
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status} from ${url}`)
      const buf = await resp.arrayBuffer()
      if (buf.byteLength > 500 * 1024 * 1024) {
        throw new Error(`Model too large for browser (${(buf.byteLength / 1024 / 1024).toFixed(0)}MB). Max: 500MB`)
      }
      devDebug('Model fetched', { size_bytes: buf.byteLength })
      const arch = inferArch(buf)
      devDebug('Inferred architecture', arch)

      if (arch.archType === 'transformer') {
        const engine = new SoulTransformerWebGPU()
        await engine.init()
        const numLayers = arch.numLayers
        const embedDim = arch.embedDim
        const dimFF = Math.round(Math.sqrt((new Float32Array(buf, 0, 1)).length)) > 0 ? 1024 : 1024
        await engine.load(buf, {
          archType: 'transformer',
          embedDim,
          numHeads: 8,
          numKVHeads: 8,
          numLayers,
          dimFF,
          vocabSize: arch.vocabSize,
          maxSeqLen: 2048,
          eps: 1e-5,
        })
        engineRef.current = engine
        setLocalArchInfo(`${embedDim}×${numLayers}×8 Transformer`)
        showToast(`Transformer engine ready (${embedDim}×${numLayers}L)`)
      } else {
        const engine = new SoulNetWebGPU()
        await engine.init()
        await engine.load(buf, { ...arch })
        engineRef.current = engine
        setLocalArchInfo(`${arch.embedDim}×${arch.hiddenDim} LSTM`)
        showToast(`Local engine ready (${arch.embedDim}x${arch.hiddenDim})`)
      }
      return true
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'unknown error'
      showToast(`Failed to load local engine: ${msg}`, 'error')
      devDebug('Local engine init failed', { error: msg, url: localModelUrl })
      return false
    } finally {
      engineLoadingRef.current = false
      setLocalEngineLoading(false)
    }
  }, [showToast, localModelUrl])

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
        ...session,
        id: newId,
        name: `${session.name} (copy)`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        starred: false,
        pinned: false,
      }
      await chatDB.saveSession(newSession)
      const newSessions = await chatDB.loadSessions()
      setSessions(newSessions)
      showToast('Conversation duplicated')
    }
  }, [showToast])

  // Only auto-save if session has received a successful response
  useEffect(() => {
    if (messages.length > 0 && sessionSaved) {
      const timeout = setTimeout(saveSession, 1000)
      return () => clearTimeout(timeout)
    }
  }, [messages, saveSession, sessionSaved])

  // Load session on mount
  useEffect(() => {
    const currentId = localStorage.getItem(CURRENT_SESSION_KEY)
    if (currentId) {
      loadSession(currentId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleCopy = useCallback((text: string) => {
    showToast('Copied to clipboard')
  }, [showToast])

  const handleRegenerate = useCallback(async () => {
    const currentMessages = messagesRef.current
    if (currentMessages.length < 2) return

    const lastAssistantIdx = currentMessages.findLastIndex(m => m.role === 'assistant')
    if (lastAssistantIdx <= 0) return

    const contextMessages = currentMessages.slice(0, lastAssistantIdx + 1)
    const assistantId = currentMessages[lastAssistantIdx].id

    setMessages(prev => prev.map(msg =>
      msg.id === assistantId
        ? { ...msg, content: '', timestamp: new Date() }
        : msg
    ))
    setLoading(true)

    try {
      for await (const data of chatController.regenerateStream(
        sessionIdRef.current,
        contextMessages.map(m => ({ role: m.role, content: m.content }))
      )) {
        if (data.error) {
          showToast('Regeneration failed', 'error')
          break
        }
        if (data.token) {
          setMessages(prev => {
            const updated = prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, content: (msg.content || '') + data.token }
                : msg
            )
            messagesRef.current = updated
            return updated
          })
        }
        if (data.done) {
          break
        }
      }
    } catch (err) {
      console.error('Regenerate error:', err)
    } finally {
      setLoading(false)
      // Store the regenerated messages (not the old contextMessages)
      storeSessionContext(sessionIdRef.current, messagesRef.current)
    }
  }, [showToast])
  handleRegenerateRef.current = handleRegenerate

  const storeSessionContext = async (sessionId: string, msgs: ChatMessage[]) => {
    try {
      await chatController.saveSessionContext(sessionId, msgs.map(m => ({ role: m.role, content: m.content })))
    } catch {}
  }

  const handleThumbsUp = useCallback(async (messageId: string) => {
    const allMsgs = messagesRef.current
    const msgIdx = allMsgs.findIndex(m => m.id === messageId)
    const assistantMsg = allMsgs[msgIdx]
    const userMsg = msgIdx > 0 ? allMsgs[msgIdx - 1] : null

    const success = await recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_up',
      conversationId: sessionIdRef.current,
      userId: userIdRef.current,
    })
    
    if (success) {
      showToast('Thanks for the feedback!')
    } else {
      showToast('Failed to submit feedback', 'error')
    }
  }, [showToast, recordFeedback])

  const handleThumbsDown = useCallback(async (messageId: string) => {
    const allMsgs = messagesRef.current
    const msgIdx = allMsgs.findIndex(m => m.id === messageId)
    const assistantMsg = allMsgs[msgIdx]
    const userMsg = msgIdx > 0 ? allMsgs[msgIdx - 1] : null

    const success = await recordFeedback({
      userMessage: userMsg?.content || '',
      assistantResponse: assistantMsg?.content || '',
      rating: 'thumbs_down',
      conversationId: sessionIdRef.current,
      userId: userIdRef.current,
    })

    if (success) {
      showToast('Thanks for the feedback!')
    } else {
      showToast('Failed to submit feedback', 'error')
    }
  }, [showToast, recordFeedback])

  const handleEditMessage = useCallback((messageId: string, newContent: string) => {
    const allMsgs = messagesRef.current
    const msgIndex = allMsgs.findIndex(m => m.id === messageId)
    if (msgIndex === -1) return
    
    // Keep messages BEFORE the edited message (not including it)
    const keepCount = msgIndex
    
    const sliced = allMsgs.slice(0, keepCount)
    messagesRef.current = sliced
    setMessages(sliced)
    setLoading(false)
    
    setCurrentError(null)
    
    // sendMessage is declared below; call via ref to avoid ordering issue
    setTimeout(() => sendMessage(newContent), 0)
  }, [])

  const handleRetry = () => {
    setCurrentError(null)
    // Re-send the last user message if input is empty (typical after error)
    const lastUser = messagesRef.current.findLast(m => m.role === 'user')
    if (lastUser?.content) {
      sendMessage(lastUser.content)
    } else if (input.trim()) {
      sendMessage()
    }
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (loading && loadingRef.current) {
          loadingRef.current.abort()
          setLoading(false)
          setStreamingStats(undefined)
        } else if (currentError) {
          setCurrentError(null)
        } else if (showSettings) {
          setShowSettings(false)
        }
      }
      if (e.key === '?' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setShowSettings(prev => !prev)
      }
      if (e.key === 'n' && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault()
        newChatRef.current()
      }
      if (e.key === 'r' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        handleRegenerateRef.current()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentError, showSettings, loading])

  const handleAddImage = useCallback((dataUrl: string) => {
    const newImage: ImageAttachment = {
      id: Date.now().toString(),
      dataUrl,
      name: `image-${Date.now()}.png`,
    }
    setImages(prev => [...prev, newImage])
    multimodalController.trainImage(dataUrl, newImage.name).then((res) => {
      devDebug('Vision trained on uploaded image', res.caption)
      multimodalController.getCapabilities().then(setVisionCaps).catch(() => {})
      multimodalController.getTrainingReport().then(r => {
        setVisionCaptionHistory(r.caption_history || [])
        setVisionVocabSize(r.vocab_size)
      }).catch(() => {})
    }).catch(() => {})
  }, [])

  const handleRemoveImage = useCallback((id: string) => {
    setImages(prev => prev.filter(img => img.id !== id))
  }, [])

  const sendMessage = async (overrideText?: string) => {
    const text = overrideText ?? input
    if ((!text.trim() && images.length === 0) || loading) return
    
    const userImages = [...images]
    
    // Get custom context from settings
    let customContext = ''
    try {
      const settingsStored = localStorage.getItem('sloughgpt_settings')
      if (settingsStored) {
        const settings = JSON.parse(settingsStored)
        customContext = settings.customContext || ''
      }
    } catch {
      // ignore
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text.trim(), // Pure user text, no knowledge injected into UI
      timestamp: new Date(),
      images: userImages.length > 0 ? userImages : undefined,
    }
    
    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }
    
    setMessages(prev => [...prev, userMessage, assistantMessage])
    setInput('')
    setImages([])
    setCurrentError(null)
    setLoading(true)
    streamStartRef.current = Date.now()
    setStreamingStats({ tokens: 0, timeElapsed: 0, tokensPerSecond: 0 })
    
    devDebug('Sending message', { model, maxTokens, temperature, messageCount: messages.length + 1 })

    // Build system prompt: Model + Soul + Agent
    // Model = neural network (base capabilities)
    // Soul = personality traits (emotional/behavioral style)
    // Agent = role/expertise (task-specific instructions)
    const parts: string[] = []
    
    // Soul provides personality context
    if (currentSoul) {
      parts.push(`[Personality: ${currentSoul.name}]`)
      if (currentSoul.description) parts.push(currentSoul.description)
      if (currentSoul.traits && currentSoul.traits.length > 0) {
        parts.push(`Traits: ${currentSoul.traits.join(', ')}`)
      }
    }
    
    // Agent provides role and expertise
    if (currentAgent) {
      parts.push(`[Role: ${currentAgent.name}]`)
      if (currentAgent.description) parts.push(currentAgent.description)
      if (currentAgent.instructions) parts.push(currentAgent.instructions)
    }
    
    const systemPrompt = parts.join('\n\n')
    
    // Save to IndexedDB immediately (crash recovery)
    const messagesWithNew = [...messagesRef.current, userMessage, assistantMessage]
    saveSessionToStorage(messagesWithNew, sessionIdRef.current).catch(console.error)
    messagesRef.current = messagesWithNew

    // Create abort controller for stopping
    loadingRef.current = new AbortController()

    // Auto-init local engine if toggled but not loaded
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
          const elapsed = (now - streamStartRef.current) / 1000
          setStreamingStats({
            tokens: assistantContentLen,
            timeElapsed: Math.floor(elapsed),
            tokensPerSecond: elapsed > 0 ? Number((assistantContentLen / elapsed).toFixed(1)) : 0,
          })
        }
        if (!hasContent) {
          setMessages(prev => prev.map(msg =>
            msg.id === assistantId ? { ...msg, content: '(empty response)' } : msg
          ))
        }
        setSessionSaved(true)
      } else {
      const response = await fetch(API_CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messagesWithNew.map(m => ({ role: m.role, content: m.content })),
          model,
          system_prompt: systemPrompt,
          max_new_tokens: maxTokens,
          temperature,
          user_id: userIdRef.current,
          session_id: sessionIdRef.current,
          images: userImages.length > 0 ? userImages.map((img: { dataUrl: string }) => img.dataUrl) : undefined,
        }),
        signal: loadingRef.current.signal,
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => '')
        devDebug('API error', { status: response.status, errorText })
        setCurrentError(getErrorInfo(response.status, errorText))
        setMessages(prev => prev.filter(msg => msg.id !== assistantId))
        setLoading(false)
        setStreamingStats(undefined)
        return
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let hasContent = false
      let assistantContentLen = 0
      
      if (reader) {
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          
          for (const line of lines) {
            const trimmed = line.trimEnd()
            if (!trimmed.startsWith('data:')) continue
            const payload = trimmed.slice(5).trim()
            if (!payload || payload === '[DONE]') continue
            try {
              const envelope = JSON.parse(payload) as {
                stream?: string; phase?: string; status?: string;
                data?: Record<string, unknown>;
                error?: string; message?: string;
              }
              if (envelope.status === 'error') {
                const errStr = typeof envelope.data?.error === 'string' ? envelope.data.error : undefined
                setCurrentError(getErrorInfo(500, envelope.message || errStr || 'Stream error'))
                setMessages(prev => prev.filter(m => m.id !== assistantId))
                setStreamingStats(undefined)
                return
              }
              const d = envelope.data ?? {}
              if (d.source && typeof d.source === 'string') {
                const fc = typeof d.fact_count === 'number' ? d.fact_count : 0
                showToast(`Knowledge: ${fc} facts from ${d.source}`, 'info')
              }
              const token = d.token as string | undefined
              if (token) {
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
                let needsSave = shouldSave
                setMessages(prev => {
                  const updated = prev.map(m =>
                    m.id === assistantId
                      ? { ...m, content: m.content + cleanedToken }
                      : m
                  )
                  messagesRef.current = updated
                  if (needsSave) {
                    saveSessionToStorage(updated, sessionIdRef.current).catch(console.error)
                  }
                  return updated
                })
                const elapsed = (now - streamStartRef.current) / 1000
                setStreamingStats({
                  tokens: assistantContentLen,
                  timeElapsed: Math.floor(elapsed),
                  tokensPerSecond: elapsed > 0 ? Number((assistantContentLen / elapsed).toFixed(1)) : 0
                })
              }
              if (envelope.status === 'complete') break
            } catch {}
          }
        }
      }

      if (!hasContent) {
        setMessages(prev => prev.map(msg => 
          msg.id === assistantId 
            ? { ...msg, content: '(empty response)' }
            : msg
        ))
      }
      setSessionSaved(true)
      storeSessionContext(sessionIdRef.current, messagesRef.current).catch(console.error)
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // User stopped the stream - keep partial content
        devDebug('Stream aborted by user')
      } else {
        setCurrentError(getErrorInfo(0, err instanceof Error ? err.message : 'Network error'))
        setMessages(prev => prev.filter(msg => msg.id !== assistantId))
      }
    } finally {
      setLoading(false)
      setStreamingStats(undefined)
      loadingRef.current = null
    }
  }

  const clearChat = useCallback(() => {
    newChat()
  }, [newChat])

  const handleSuggestionClick = useCallback((text: string) => {
    setInput(text)
    sendMessage(text)
  }, [sendMessage])

  const toggleSettings = useCallback(() => {
    setShowSettings(prev => !prev)
  }, [])

const sidebarConversations: Conversation[] = (Array.isArray(sessions) ? sessions : []).map(s => ({
    id: s.id,
    name: s.name || 'Untitled',
    session_id: s.id,
    created_at: s.createdAt || new Date().toISOString(),
    updated_at: s.updatedAt || new Date().toISOString(),
    pinned: Boolean(s.pinned),
    starred: Boolean(s.starred),
    message_count: Array.isArray(s.messages) ? s.messages.length : 0,
    messages: (Array.isArray(s.messages) ? s.messages : []).map(m => ({
      id: String(m.id || `msg_${Date.now()}`),
      role: String(m.role || 'user'),
      content: String(m.content || ''),
      timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : String(m.timestamp || Date.now()),
    })),
    synced: Boolean(s.synced),
  }))

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">

      {/* Unified chat box — everything except nav sidebar */}
      <div className="flex flex-1 min-h-0 overflow-hidden rounded-lg border border-border/50 bg-background shadow-sm hover:shadow-md transition-shadow duration-300">

        <ChatSidebar
          conversations={sidebarConversations}
          currentConversationId={sessionIdRef.current}
          onLoadConversation={loadSession}
          onDeleteConversation={deleteSession}
          onStarConversation={starSession}
          onPinConversation={pinSession}
          onRenameConversation={renameSession}
          onDuplicateConversation={duplicateSession}
          onNewChat={newChat}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(prev => !prev)}
          onOpenConversationSearch={() => setShowConversationSearch(true)}
        />

        <div className="flex flex-col flex-1 min-h-0 min-w-0 max-w-full overflow-hidden">
          {/* Chat header */}
          <div className="lg:sticky lg:top-0 z-10 flex items-center justify-end lg:justify-center px-3 py-2 border-b border-border/40 shrink-0 bg-background/80 backdrop-blur-sm gap-2">
            {/* Search */}
            <div className="flex items-center gap-2 min-w-0">
              <div className="relative w-36 sm:w-44">
                <IconSearch className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/60 pointer-events-none" aria-hidden />
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => { setSearchQuery(e.target.value); setMatchIndex(0) }}
                  className="w-full pl-7 pr-6 py-1.5 text-xs rounded-lg border border-input bg-background/80 focus:outline-none focus:ring-1 focus:ring-primary/40 placeholder:text-muted-foreground/50 transition-shadow hover:shadow-sm"
                  aria-label="Search messages"
                />
                {searchQuery && (
                  <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
                    {matchCount > 0 && (
                      <span className="text-[10px] text-muted-foreground whitespace-nowrap mr-0.5">
                        {matchIndex + 1}/{matchCount}
                      </span>
                    )}
                    <button
                      onClick={() => {
                        const newIdx = matchIndex > 0 ? matchIndex - 1 : matchCount - 1
                        setMatchIndex(newIdx)
                        const el = document.getElementById(`msg-${matchIds[newIdx]}`)
                        el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                      }}
                      className="text-muted-foreground hover:text-foreground disabled:opacity-30 p-0.5"
                      disabled={matchCount === 0}
                      aria-label="Previous match"
                    >
                      <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg>
                    </button>
                    <button
                      onClick={() => {
                        const newIdx = matchIndex < matchCount - 1 ? matchIndex + 1 : 0
                        setMatchIndex(newIdx)
                        const el = document.getElementById(`msg-${matchIds[newIdx]}`)
                        el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                      }}
                      className="text-muted-foreground hover:text-foreground disabled:opacity-30 p-0.5"
                      disabled={matchCount === 0}
                      aria-label="Next match"
                    >
                      <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                    </button>
                    <button
                      onClick={() => setSearchQuery('')}
                      className="text-muted-foreground hover:text-foreground transition-colors p-0.5"
                      aria-label="Clear search"
                    >
                      <IconX className="h-3 w-3" />
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1 sm:gap-1.5">
              {/* Model + status */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 px-2.5 font-mono text-xs gap-1.5 rounded-lg border border-transparent hover:border-border/50">
                    <span className={`inline-block h-1.5 w-1.5 rounded-full shrink-0 ${
                      loadingModel ? 'bg-warning animate-pulse' :
                      model ? (loading ? 'bg-warning animate-pulse' : 'bg-success') :
                      'bg-muted-foreground/30'
                    }`} />
                    <span className="truncate max-w-[48px] sm:max-w-[64px]" title={loadingModel || model || 'Select a model to load'}>
                      {loadingModel
                        ? (loadingModel.includes('/') ? loadingModel.split('/').pop() : loadingModel)
                        : model
                          ? (model.includes('/') ? model.split('/').pop() : model)
                          : 'Select model'}
                    </span>
                    <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-[200px] max-h-[300px] overflow-y-auto">
                  {loadingModel && (
                    <div className="h-0.5 bg-muted rounded-full mx-2 mb-1 overflow-hidden shrink-0">
                      <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: '60%' }} />
                    </div>
                  )}
                  {availableModels.map((m) => {
                    const info = modelInfoMap[m]
                    const isCached = info?.cached
                    const sizeLabel = info?.size_gb ? `${info.size_gb.toFixed(2)} GB` : ''
                    const isLoaded = m === model
                    const isLoading = m === loadingModel
                    return (
                      <DropdownMenuItem
                        key={m}
                        onSelect={async () => {
                          if (m === model || loadingModel) return
                          setLoadingModel(m)
                          if (isCached) {
                            showToast(`Loading ${m}...`, 'info')
                            try {
                              const result = await modelController.load(m)
                              await refreshHealth()
                              setModel(m)
                              showToast(`Model ready: ${m} (${result.device || 'cpu'})`, 'success')
                            } catch (err) {
                              showToast(`Failed to load ${m}: ${err instanceof Error ? err.message : 'unknown error'}`, 'error')
                            } finally {
                              setLoadingModel(null)
                            }
                          } else {
                            // Confirm before downloading uncached model
                            const sizeText = info?.size_gb ? `${info.size_gb.toFixed(1)} GB` : '? GB'
                            const modelName = m.includes('/') ? m.split('/').pop() : m
                            if (!window.confirm(`Download ${modelName} (${sizeText}) from HuggingFace?`)) {
                              setLoadingModel(null); return
                            }
                            showToast(`Downloading ${m}...`, 'info')
                            try {
                              await startDownload(m, info?.size_gb ? Math.round(info.size_gb * 1024 * 1024 * 1024) : 0)
                              await modelController.load(m)
                              await refreshHealth()
                              setModel(m)
                              showToast(`Model ready: ${m}`, 'success')
                            } catch (err) {
                              showToast(`Failed: ${err instanceof Error ? err.message : 'unknown'}`, 'error')
                            } finally {
                              setLoadingModel(null)
                            }
                          }
                        }}
                        disabled={isLoading}
                        className="font-mono text-xs"
                        title={`${m}${isCached ? ' (cached)' : ' (download)'}${sizeLabel ? ` — ${sizeLabel}` : ''}`}
                      >
                        <span className="truncate flex-1">{m.includes('/') ? m.split('/').pop() : m}</span>
                        <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sizeLabel}</span>
                        {isLoading ? (
                          <IconRefresh className="h-3 w-3 animate-spin shrink-0 text-warning ml-1" />
                        ) : isLoaded ? (
                          <IconCheck className="h-3 w-3 shrink-0 text-success ml-1" />
                        ) : isCached ? (
                          <span className="text-[9px] text-muted-foreground/40 px-1 ml-1 border border-border/30 rounded leading-none">cached</span>
                        ) : (
                          <svg className="h-2.5 w-2.5 shrink-0 text-muted-foreground/40 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                        )}
                      </DropdownMenuItem>
                    )
                  })}
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Soul pill (personality) */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-full bg-primary/8 text-primary hover:bg-primary/15 border border-primary/15" title={currentSoul?.traits?.join(', ')}>
                    <IconHeart className="h-3 w-3 shrink-0" />
                    <span className="truncate max-w-[48px] sm:max-w-[64px]">{currentSoul?.name || 'Personality'}</span>
                    <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-[160px]">
                  {souls.map((s) => (
                    <DropdownMenuItem
                      key={s.name}
                      onSelect={async () => {
                        try { await soulsController.switch(s.name); setCurrentSoul(s) }
                        catch (e) { console.error('Failed to switch soul:', e) }}
                      }
                      className="justify-between text-xs"
                    >
                      <span>{s.name}</span>
                      {currentSoul?.name === s.name && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Agent selector (role/expertise) */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg border border-border/40 hover:border-border/60" title={currentAgent?.description}>
                    <span className="text-sm shrink-0">{currentAgent?.id === 'coder' ? '💻' : currentAgent?.id === 'writer' ? '✍️' : currentAgent?.id === 'researcher' ? '🔬' : currentAgent?.id === 'analyst' ? '📊' : '💬'}</span>
                    <span className="truncate max-w-[48px] sm:max-w-[64px]">{currentAgent?.name || 'Role'}</span>
                    <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-[180px]">
                  {agents.map((a) => (
                    <DropdownMenuItem
                      key={a.id}
                      onSelect={() => {
                        setCurrentAgent({ id: a.id, name: a.name, description: a.description || '', instructions: a.instructions || '' })
                        localStorage.setItem('sloughgpt_current_agent', a.id)
                        showToast(`Switched to ${a.name}`, 'info')
                      }}
                      className="justify-between text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm">{a.id === 'coder' ? '💻' : a.id === 'writer' ? '✍️' : a.id === 'researcher' ? '🔬' : a.id === 'analyst' ? '📊' : '💬'}</span>
                        <div>
                          <div>{a.name}</div>
                          <div className="text-[10px] text-muted-foreground">{a.description}</div>
                        </div>
                      </div>
                      {currentAgent?.id === a.id && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Server/Local toggle — only show if a .soul URL is configured */}
              {localModelUrl && (
              <Button
                variant={useLocalEngine ? 'default' : 'ghost'}
                size="sm"
                className="h-7 px-2.5 text-xs gap-1.5 rounded-lg"
                disabled={localEngineLoading}
                onClick={async () => {
                  if (useLocalEngine) {
                    setUseLocalEngine(false)
                    setLocalArchInfo(null)
                    showToast('Switched to server inference', 'info')
                    devDebug('Switched to server mode')
                  } else if (engineRef.current) {
                    setUseLocalEngine(true)
                    showToast(`Local GPU ready (${localArchInfo})`, 'success')
                    devDebug('Switched to local mode', { arch: localArchInfo })
                  } else {
                    showToast('Loading local engine...', 'info')
                    devDebug('Attempting local engine init', { url: localModelUrl })
                    if (await initLocalEngine()) {
                      setUseLocalEngine(true)
                      showToast(`Local GPU ready (${localArchInfo})`, 'success')
                    } else {
                      showToast('Local engine failed — check .soul file URL', 'error')
                      devDebug('Local engine init failed')
                    }
                  }
                }}
                title={localEngineLoading ? 'Loading local engine...' : localArchInfo ? `Local GPU (${localArchInfo})` : useLocalEngine ? 'Running locally on GPU' : 'Running on server'}
                aria-pressed={useLocalEngine}
              >
                {localEngineLoading ? (
                  <IconRefresh className="h-3 w-3 animate-spin" />
                ) : useLocalEngine ? <Cpu className="h-3 w-3" /> : <Server className="h-3 w-3" />}
                <span className="hidden sm:inline">{localEngineLoading ? 'Loading' : useLocalEngine ? 'Local' : 'Server'}</span>
              </Button>
              )}

              {/* More menu (Export + Tools) */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 px-2.5 text-xs gap-1.5 rounded-lg" aria-label="More options">
                    <IconMore className="h-3.5 w-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-[160px]">
                  <DropdownMenuItem onSelect={() => setVoiceMode(true)}>
                    <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                      <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                    </svg>
                    Voice Mode
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setToolPanelOpen(prev => !prev)}>
                    <IconSettings className="mr-2 h-4 w-4" />
                    Tools Panel
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => exportConversationAsMarkdown(messages)} disabled={messages.length === 0}>
                    <IconDownload className="mr-2 h-4 w-4" />
                    Export Markdown
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          <ChatSettings
          isOpen={showSettings}
          model={model}
          temperature={temperature}
          maxTokens={maxTokens}
          onModelChange={setModel}
          availableModels={availableModels}
          onTemperatureChange={(temp) => {
            setTemperature(temp)
            generationConfigController.update({ temperature: temp }).catch(() => {})
          }}
          onMaxTokensChange={(tokens) => {
            setMaxTokens(tokens)
            generationConfigController.update({ max_new_tokens: tokens }).catch(() => {})
          }}
          onClear={clearChat}
          hasMessages={messages.length > 0}
        />

        {currentError && (
          <ErrorBanner
            error={currentError}
            onRetry={handleRetry}
            onDismiss={() => setCurrentError(null)}
          />
        )}

        <ChatArea
          messages={messages}
          loading={loading}
          health={health}
          onRefreshHealth={refreshHealth}
          onCopy={handleCopy}
          onRegenerate={handleRegenerate}
          onThumbsUp={handleThumbsUp}
          onThumbsDown={handleThumbsDown}
          onEdit={handleEditMessage}
          searchQuery={searchQuery}
          onSuggestionClick={handleSuggestionClick}
          value={input}
          onChange={setInput}
          onSend={sendMessage}
          onStop={() => {
            if (loadingRef.current) {
              loadingRef.current.abort()
            }
            setLoading(false)
            setStreamingStats(undefined)
          }}
          images={images}
          onAddImage={handleAddImage}
          onRemoveImage={handleRemoveImage}
          streamingStats={streamingStats}
        />

        <ConversationViewer
          isOpen={showConversationViewer}
          onClose={() => setShowConversationViewer(false)}
          messages={messages.map(m => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp: typeof m.timestamp === 'number' ? m.timestamp : m.timestamp?.getTime() || Date.now(),
          }))}
          title="Current Conversation"
        />

        <ConversationSearch
          open={showConversationSearch}
          onClose={() => setShowConversationSearch(false)}
          onNavigate={(sessionId) => loadSession(sessionId)}
        />

        </div>
      </div>

      <ChatToolPanel
        open={toolPanelOpen}
        onClose={() => setToolPanelOpen(false)}
        learnerInfo={learnerInfo}
        learnerTraining={learnerTraining}
        onTrainStep={async () => {
          setLearnerTraining(true)
          try {
            const resp = await fetch(`${PUBLIC_API_URL}/learn/train`, { method: 'POST' })
            if (resp.ok) {
              const data = await resp.json()
              if (data.current_loss !== undefined) showToast(`Train step: loss ${data.current_loss.toFixed(4)}`)
              else showToast('Train step complete')
              setLearnerInfo(prev => {
                if (!prev) return prev
                return {
                  ...prev,
                  train_steps_completed: data.train_steps_completed ?? prev.train_steps_completed,
                  current_loss: data.current_loss ?? prev.current_loss,
                  loss_history: data.loss_history ?? prev.loss_history,
                }
              })
            } else showToast('Train step failed', 'error')
          } catch { showToast('Train step failed', 'error') }
          finally { setLearnerTraining(false) }
        }}
        checkpoints={checkpoints}
        currentCheckpoint={currentCheckpoint}
        onLoadCheckpoint={async (name) => {
          try {
            await soulsController.loadCheckpoint(name)
            setCurrentCheckpoint(name)
            showToast(`Checkpoint loaded: ${name}`)
          } catch {
            showToast('Failed to load checkpoint', 'error')
          }
        }}
        agents={agents}
        currentAgent={currentAgent}
        onSelectAgent={setCurrentAgent}
        availableModels={availableModels}
        currentModel={model}
        onSelectModel={async (m) => {
          if (m === model) return
          setModel(m)
          try {
            await modelController.load(m)
            showToast(`Model loaded: ${m}`)
          } catch {
            showToast(`Failed to load model: ${m}`, 'error')
          }
        }}
        souls={souls}
        currentSoulName={currentSoul?.name}
        onSwitchSoul={async (name) => {
          try {
            await soulsController.switch(name)
            const s = souls.find(s => s.name === name)
            if (s) setCurrentSoul(s)
          } catch (e) {
            console.error('Failed to switch soul:', e)
          }
        }}
        onOpenSettings={toggleSettings}
        onOpenShortcuts={() => window.dispatchEvent(new CustomEvent('toggle-shortcuts'))}
        onOpenConversationViewer={() => setShowConversationViewer(true)}
        visionImagesLearned={visionCaps?.images_learned}
        visionTrained={visionCaps?.trained}
        visionStatus={visionCaps?.status}
        visionCaptionHistory={visionCaptionHistory}
        visionVocabSize={visionVocabSize}
      />

      {/* Voice Chat Mode overlay */}
      {voiceMode && (
        <VoiceChatMode
          onMessage={async (text) => {
            setInput(text)
            await sendMessage(text)
          }}
          onClose={() => setVoiceMode(false)}
        />
      )}
    </div>
  )
}

function buildLocalPrompt(messages: ChatMessage[], systemPrompt: string): string {
  let prompt = systemPrompt ? `System: ${systemPrompt}\n` : ''
  for (const m of messages) {
    if (m.role === 'user') prompt += `User: ${m.content}\n`
    else prompt += `Assistant: ${m.content}\n`
  }
  prompt += 'Assistant:'
  return prompt
}

function exportConversationAsMarkdown(messages: ChatMessage[]) {
  const lines: string[] = [
    '# Chat Conversation',
    `*Exported on ${new Date().toLocaleString()}*`,
    '---',
    '',
  ]
  for (const m of messages) {
    const role = m.role === 'user' ? '**You**' : '**Assistant**'
    const timestamp = m.timestamp
      ? new Date(typeof m.timestamp === 'number' ? m.timestamp : m.timestamp).toLocaleTimeString()
      : ''
    lines.push(`### ${role} ${timestamp ? `(${timestamp})` : ''}`)
    lines.push('')
    lines.push(m.content)
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `chat-export-${Date.now()}.md`
  a.click()
  URL.revokeObjectURL(url)
}
