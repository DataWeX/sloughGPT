import {create} from 'zustand';
import {api} from '../services/api-client';
import {streamSSE} from '../services/sse-client';
import {useSettingsStore} from './settings-store';
import type {Message, Session} from '../types';

interface ChatState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  streaming: boolean;
  error: string | null;
  refreshSessions: () => Promise<void>;
  loadSession: (id: string) => Promise<void>;
  createSession: () => Promise<string>;
  deleteSession: (id: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  regenerate: (messageId: string) => Promise<void>;
  cancelStream: () => void;
  recordFeedback: (messageId: string, positive: boolean) => Promise<void>;
  clearError: () => void;
}

let abortController: AbortController | null = null;

function buildProviderMessages(messages: Message[]) {
  return messages.map(m => ({role: m.role, content: m.content}));
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  streaming: false,
  error: null,

  refreshSessions: async () => {
    try {
      const sessions = await api.get<Session[]>('/chat/sessions');
      set({sessions});
    } catch {}
  },

  loadSession: async (id: string) => {
    try {
      const data = await api.get<{messages: Message[]}>(
        `/session/${id}/messages`,
      );
      set({activeSessionId: id, messages: data.messages || []});
    } catch (err: any) {
      set({error: err.message});
    }
  },

  createSession: async () => {
    try {
      const result = await api.post<{id: string}>('/chat/sessions');
      const {id} = result;
      set({activeSessionId: id, messages: []});
      await get().refreshSessions();
      return id;
    } catch (err: any) {
      set({error: err.message});
      return '';
    }
  },

  deleteSession: async (id: string) => {
    try {
      await api.delete(`/chat/sessions/${id}`);
      if (get().activeSessionId === id) {
        set({activeSessionId: null, messages: []});
      }
      await get().refreshSessions();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  sendMessage: async (content: string) => {
    const state = get();
    let sessionId = state.activeSessionId;

    if (!sessionId) {
      sessionId = await get().createSession();
      if (!sessionId) return;
    }

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    const assistantMsg: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    set({
      messages: [...state.messages, userMsg, assistantMsg],
      streaming: true,
      error: null,
    });

    abortController = new AbortController();
    let accumulated = '';

    const settings = useSettingsStore.getState();

    try {
      for await (const event of streamSSE(
        '/chat/stream',
        {
          messages: [...state.messages, userMsg].map(m => ({
            role: m.role,
            content: m.content,
          })),
          temperature: settings.temperature,
          max_new_tokens: settings.maxTokens,
        },
        abortController.signal,
      )) {
        if (event.token) {
          accumulated += event.token;
          set(s => ({
            messages: s.messages.map(m =>
              m.id === assistantMsg.id ? {...m, content: accumulated} : m,
            ),
          }));
        }
        if (event.error) {
          set({error: event.error, streaming: false});
          return;
        }
        if (event.done) break;
      }

      api
        .post(`/session/${sessionId}/context`, {
          messages: [...state.messages, userMsg, {...assistantMsg, content: accumulated}].map(m => ({
            role: m.role,
            content: m.content,
          })),
        })
        .catch(() => {});

      await get().refreshSessions();
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      set({error: err.message});
    } finally {
      set({streaming: false});
      abortController = null;
    }
  },

  regenerate: async (messageId: string) => {
    const state = get();
    const sessionId = state.activeSessionId;
    if (!sessionId) return;

    const msgIndex = state.messages.findIndex(m => m.id === messageId);
    if (msgIndex < 0) return;

    const contextMessages = state.messages.slice(0, msgIndex).map(m => ({
      role: m.role,
      content: m.content,
    }));

    const assistantMsg: Message = {
      id: `regen-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    set({
      messages: [...state.messages.slice(0, msgIndex), assistantMsg],
      streaming: true,
      error: null,
    });

    abortController = new AbortController();
    let accumulated = '';

    try {
      for await (const event of streamSSE(
        `/session/${sessionId}/regenerate`,
        {messages: contextMessages},
        abortController.signal,
      )) {
        if (event.token) {
          accumulated += event.token;
          set(s => ({
            messages: s.messages.map(m =>
              m.id === assistantMsg.id ? {...m, content: accumulated} : m,
            ),
          }));
        }
        if (event.done) break;
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      set({error: err.message});
    } finally {
      set({streaming: false});
      abortController = null;
    }
  },

  cancelStream: () => {
    abortController?.abort();
    set({streaming: false});
  },

  recordFeedback: async (messageId: string, positive: boolean) => {
    try {
      await api.post('/feedback/workflow-record', {
        message_id: messageId,
        positive,
        session_id: get().activeSessionId,
      });
    } catch {}
  },

  clearError: () => set({error: null}),
}));
