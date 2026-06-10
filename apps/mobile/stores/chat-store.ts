import { create } from 'zustand'
import { apiGet, apiPost, apiPut, apiDelete } from '@/lib/api-client'
import { streamSSE, createAbortController } from '@/lib/sse-client'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  created_at: string
  updated_at: string
  starred?: boolean
  pinned?: boolean
}

interface ChatState {
  sessions: Conversation[]
  activeSessionId: string | null
  messages: Message[]
  streaming: boolean
  streamBuffer: string
  error: string | null

  refreshSessions: () => Promise<void>
  loadSession: (id: string) => Promise<void>
  createSession: () => Promise<string>
  deleteSession: (id: string) => Promise<void>
  updateSession: (id: string, data: { title?: string; starred?: boolean; pinned?: boolean }) => Promise<void>

  sendMessage: (content: string) => Promise<void>
  regenerate: (messageId: string) => Promise<void>
  cancelStream: () => void

  recordFeedback: (messageId: string, positive: boolean) => Promise<void>
  clearError: () => void
}

let abortController: AbortController | null = null

function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

export const useChatStore = create<ChatState>()((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  streaming: false,
  streamBuffer: '',
  error: null,

  refreshSessions: async () => {
    try {
      const data = await apiGet<{ sessions: Conversation[] }>('/chat/sessions')
      set({ sessions: data.sessions || [] })
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  loadSession: async (id: string) => {
    try {
      const data = await apiGet<{ messages: Message[] }>(`/session/${id}/messages`)
      set({
        activeSessionId: id,
        messages: data.messages || [],
        error: null,
      })
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  createSession: async () => {
    try {
      const data = await apiPost<{ session_id: string }>('/chat/sessions', {})
      const id = data.session_id
      set({
        activeSessionId: id,
        messages: [],
      })
      await get().refreshSessions()
      return id
    } catch (error) {
      set({ error: (error as Error).message })
      return ''
    }
  },

  deleteSession: async (id: string) => {
    try {
      await apiDelete(`/chat/sessions/${id}`)
      const state = get()
      if (state.activeSessionId === id) {
        set({ activeSessionId: null, messages: [] })
      }
      await get().refreshSessions()
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  updateSession: async (id, data) => {
    try {
      await apiPut(`/chat/sessions/${id}`, data)
      await get().refreshSessions()
    } catch (error) {
      set({ error: (error as Error).message })
    }
  },

  sendMessage: async (content: string) => {
    const state = get()
    let sessionId = state.activeSessionId

    if (!sessionId) {
      sessionId = await get().createSession()
      if (!sessionId) return
    }

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }

    const assistantMessage: Message = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    }

    set((s) => ({
      messages: [...s.messages, userMessage, assistantMessage],
      streaming: true,
      streamBuffer: '',
      error: null,
    }))

    abortController = createAbortController()

    const allMessages = [...get().messages.slice(0, -1)].map((m) => ({
      role: m.role,
      content: m.content,
    }))

    try {
      let accumulated = ''

      for await (const event of streamSSE(
        '/chat/stream',
        {
          messages: allMessages,
          session_id: sessionId,
        },
        abortController.signal
      )) {
        if (event.error) {
          set({ streaming: false, error: event.error })
          return
        }

        if (event.done) {
          set((s) => ({
            streaming: false,
            streamBuffer: '',
            messages: s.messages.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: accumulated }
                : m
            ),
          }))

          try {
            await apiPost(`/session/${sessionId}/context`, {
              messages: get().messages.map((m) => ({
                role: m.role,
                content: m.content,
              })),
            })
          } catch {
            // context save is best-effort
          }

          await get().refreshSessions()
          return
        }

        if (event.token) {
          accumulated += event.token
          set((s) => ({
            streamBuffer: accumulated,
            messages: s.messages.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, content: accumulated }
                : m
            ),
          }))
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        set({ streaming: false, error: (error as Error).message })
      }
    }
  },

  regenerate: async (messageId: string) => {
    const state = get()
    const sessionId = state.activeSessionId
    if (!sessionId) return

    const msgIndex = state.messages.findIndex((m) => m.id === messageId)
    if (msgIndex === -1) return

    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === messageId ? { ...m, content: '' } : m
      ),
      streaming: true,
      streamBuffer: '',
      error: null,
    }))

    abortController = createAbortController()

    const contextMessages = state.messages
      .slice(0, msgIndex)
      .map((m) => ({ role: m.role, content: m.content }))

    try {
      let accumulated = ''

      for await (const event of streamSSE(
        `/session/${sessionId}/regenerate`,
        { messages: contextMessages },
        abortController.signal
      )) {
        if (event.error) {
          set({ streaming: false, error: event.error })
          return
        }

        if (event.done) {
          set((s) => ({
            streaming: false,
            streamBuffer: '',
            messages: s.messages.map((m) =>
              m.id === messageId ? { ...m, content: accumulated } : m
            ),
          }))
          return
        }

        if (event.token) {
          accumulated += event.token
          set((s) => ({
            streamBuffer: accumulated,
            messages: s.messages.map((m) =>
              m.id === messageId ? { ...m, content: accumulated } : m
            ),
          }))
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        set({ streaming: false, error: (error as Error).message })
      }
    }
  },

  cancelStream: () => {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    set({ streaming: false, streamBuffer: '' })
  },

  recordFeedback: async (messageId: string, positive: boolean) => {
    const state = get()
    try {
      await apiPost('/feedback/workflow-record', {
        session_id: state.activeSessionId,
        message_id: messageId,
        positive,
      })
    } catch {
      // feedback is best-effort
    }
  },

  clearError: () => set({ error: null }),
}))
