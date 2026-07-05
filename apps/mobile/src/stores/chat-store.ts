import {create} from 'zustand';
import {api} from '../services/api-client';
import {streamSSE} from '../services/sse-client';
import {useSettingsStore} from './settings-store';
import {
  cacheMessages,
  getCachedMessages,
  appendCachedMessage,
  addPendingSend,
  getPendingSends,
  removePendingSend,
  cacheActiveSessionId,
} from '../services/offline-cache';
import {triggerHaptic} from '../services/haptics';
import {toast} from '../services/toast';
import type {Message, Session} from '../types';

interface ChatState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  streaming: boolean;
  error: string | null;
  offlineQueue: number;
  refreshSessions: () => Promise<void>;
  loadSession: (id: string) => Promise<void>;
  createSession: () => Promise<string>;
  deleteSession: (id: string) => Promise<void>;
  deleteMessage: (messageId: string) => void;
  sendMessage: (content: string) => Promise<void>;
  regenerate: (messageId: string) => Promise<void>;
  cancelStream: () => void;
  recordFeedback: (messageId: string, positive: boolean) => Promise<void>;
  clearError: () => void;
  retryPendingSends: () => Promise<void>;
}

let abortController: AbortController | null = null;

function genId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  streaming: false,
  error: null,
  offlineQueue: 0,

  refreshSessions: async () => {
    try {
      const sessions = await api.get<Session[]>('/chat/sessions');
      set({sessions});
    } catch {
      // offline — sessions will be stale but functional
    }
  },

  loadSession: async (id: string) => {
    try {
      const data = await api.get<{messages: Message[]}>(
        `/session/${id}/messages`,
      );
      const msgs = data.messages || [];
      set({activeSessionId: id, messages: msgs});
      await cacheActiveSessionId(id);
      await cacheMessages(id, msgs);
    } catch {
      // offline — load from cache
      const cached = await getCachedMessages(id);
      if (cached.length > 0) {
        set({activeSessionId: id, messages: cached});
      }
      set({error: 'Offline — showing cached messages'});
    }
  },

  createSession: async () => {
    try {
      const result = await api.post<{id: string}>('/chat/sessions');
      const {id} = result;
      set({activeSessionId: id, messages: []});
      await cacheActiveSessionId(id);
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
        await cacheActiveSessionId(null);
      }
      await get().refreshSessions();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  deleteMessage: (messageId: string) => {
    set(s => ({messages: s.messages.filter(m => m.id !== messageId)}));
  },

  sendMessage: async (content: string) => {
    const state = get();
    let sessionId = state.activeSessionId;

    if (!sessionId) {
      try {
        sessionId = await get().createSession();
      } catch {
        toast.error('Failed to create session. Check your connection.');
        set({error: 'Failed to create session. Check your connection.'});
        return;
      }
      if (!sessionId) return;
    }

    const userMsg: Message = {
      id: genId(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    const assistantMsg: Message = {
      id: genId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    set({
      messages: [...state.messages, userMsg, assistantMsg],
      streaming: true,
      error: null,
    });

    // Cache user message immediately
    await appendCachedMessage(sessionId, userMsg);

    abortController = new AbortController();
    let accumulated = '';
    const settings = useSettingsStore.getState();
    const body = {
      messages: [...state.messages, userMsg].map(m => ({
        role: m.role,
        content: m.content,
      })),
      temperature: settings.temperature,
      max_new_tokens: settings.maxTokens,
    };

    let retries = 0;
    const maxRetries = 2;

    const attemptStream = async (): Promise<boolean> => {
      try {
        for await (const event of streamSSE('/chat/stream', body, abortController!.signal)) {
          if (event.token) {
            accumulated += event.token;
            set(s => ({
              messages: s.messages.map(m =>
                m.id === assistantMsg.id ? {...m, content: accumulated} : m,
              ),
            }));
          }
          if (event.error) {
            if (retries < maxRetries) {
              retries++;
              await new Promise(r => setTimeout(r, 500 * retries));
              return false;
            }
            set({error: event.error, streaming: false});
            toast.error(event.error);
            await triggerHaptic('error');
            return true;
          }
          if (event.done) return true;
        }
        return true;
      } catch (err: any) {
        if (err.name === 'AbortError') return true;

        // Queue for offline retry
        await addPendingSend({
          id: userMsg.id,
          sessionId,
          content,
          timestamp: Date.now(),
          retryCount: 0,
        });
        set(s => ({offlineQueue: s.offlineQueue + 1}));

        if (retries < maxRetries) {
          retries++;
          await new Promise(r => setTimeout(r, 500 * retries));
          return false;
        }
        toast.warn('Offline — message queued for retry');
        set({error: 'Offline — message queued for retry', streaming: false});
        await triggerHaptic('error');
        return true;
      }
    };

    try {
      let done = false;
      while (!done) {
        done = await attemptStream();
      }

      // Cache completed assistant message
      if (accumulated) {
        await appendCachedMessage(sessionId, {...assistantMsg, content: accumulated});
        await removePendingSend(userMsg.id);
        await triggerHaptic('success');
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
      await triggerHaptic('error');
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
      id: genId(),
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
      if (accumulated) {
        await appendCachedMessage(sessionId, {...assistantMsg, content: accumulated});
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      toast.error('Regeneration failed');
      set({error: err.message});
      await triggerHaptic('error');
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
      await triggerHaptic(positive ? 'success' : 'light');
    } catch {
      await triggerHaptic('light');
    }
  },

  clearError: () => set({error: null}),

  retryPendingSends: async () => {
    const pending = await getPendingSends();
    if (pending.length === 0) return;

    toast.info(`Retrying ${pending.length} queued message${pending.length > 1 ? 's' : ''}...`);
    for (const send of pending) {
      set({activeSessionId: send.sessionId});
      await get().sendMessage(send.content);
    }
    set({offlineQueue: 0});
    toast.success('Messages synced');
  },
}));
