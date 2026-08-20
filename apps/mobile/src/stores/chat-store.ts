import {create} from 'zustand';
import {api} from '../services/api-client';
import {streamSSE} from '../services/sse-client';
import {useSettingsStore} from './settings-store';
import {useHybridStore} from './hybrid-inference-store';
import {useProvidersStore} from './providers-store';
import {streamProviderChat, ProviderError} from '../services/providers-client';
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
import {sounds} from '../services/sounds';
import {toast} from '../services/toast';
import {collectPair} from '../services/training-collector';
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
  archiveSession: (id: string, archived: boolean) => Promise<void>;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteMessage: (messageId: string) => void;
  sendMessage: (content: string, images?: string[]) => Promise<void>;
  forwardMessage: (content: string, targetSessionId: string) => Promise<void>;
  regenerate: (messageId: string) => Promise<void>;
  cancelStream: () => void;
  recordFeedback: (messageId: string, positive: boolean) => Promise<void>;
  clearError: () => void;
  retryPendingSends: () => Promise<void>;
}

let _abortController: AbortController | null = null;
let _everConnected = false;

function genId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

async function saveSessionContext(
  sessionId: string,
  messages: {role: string; content: string}[],
) {
  try {
    await api.post(`/session/${sessionId}/context`, {
      messages: messages.map(m => ({role: m.role, content: m.content})),
    });
  } catch {}
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
      const data = await api.get<Session[]>('/chat/sessions');
      set({sessions: data || []});
    } catch {
      // offline — sessions will be stale but functional
    }
  },

  loadSession: async (id: string) => {
    try {
      const data = await api.get<{messages: Message[]}>(
        `/session/${id}/messages`,
      );
      _everConnected = true;
      const msgs = data.messages || [];
      set({activeSessionId: id, messages: msgs});
      await cacheActiveSessionId(id);
      await cacheMessages(id, msgs);
    } catch {
      const cached = await getCachedMessages(id);
      if (cached.length > 0) {
        set({activeSessionId: id, messages: cached});
      }
      if (_everConnected) {
        set({error: 'Offline — showing cached messages'});
      }
    }
  },

  createSession: async () => {
    try {
      const result = await api.post<{session_id: string}>('/chat/sessions', {});
      const {session_id} = result;
      set({activeSessionId: session_id, messages: []});
      await cacheActiveSessionId(session_id);
      await get().refreshSessions();
      return session_id;
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

  archiveSession: async (id: string, archived: boolean) => {
    try {
      await api.archiveSession(id, archived);
      await get().refreshSessions();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  renameSession: async (id: string, name: string) => {
    try {
      await api.renameSession(id, name);
      await get().refreshSessions();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  deleteMessage: (messageId: string) => {
    set(s => ({messages: s.messages.filter(m => m.id !== messageId)}));
  },

  forwardMessage: async (content: string, targetSessionId: string) => {
    const userMsg = {
      id: genId(),
      role: 'user' as const,
      content,
      timestamp: Date.now(),
    };

    try {
      // Fetch existing session, append message, save back
      const session = await api.get<{id: string; messages: any[]}>(
        `/chat/sessions/${targetSessionId}`,
      );
      const updatedMessages = [...(session.messages || []), userMsg];
      await api.put(`/chat/sessions/${targetSessionId}`, {
        messages: updatedMessages,
      });

      // If forwarding to current session, also update local state
      if (get().activeSessionId === targetSessionId) {
        set(s => ({messages: [...s.messages, userMsg]}));
      }

      triggerHaptic('light');
      sounds.send();
      toast.success('Message forwarded');
    } catch {
      toast.error('Failed to forward message');
    }
  },

  sendMessage: async (content: string, images?: string[]) => {
    const state = get();
    let sessionId = state.activeSessionId;

    if (!sessionId) {
      try {
        sessionId = await get().createSession();
      } catch {
        toast.error('Failed to create session. Check your connection.');
        sounds.error();
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
      images: images && images.length > 0 ? images : undefined,
      status: 'sending',
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

    sounds.send();

    // Cache user message immediately
    await appendCachedMessage(sessionId, userMsg);

    // ── Shared state for all remote paths ───────────────────────────────
    const controller = new AbortController();
    _abortController = controller;
    let accumulated = '';

    const finalizeSuccess = async (text: string) => {
      set(s => ({
        messages: s.messages.map(m =>
          m.id === assistantMsg.id ? {...m, content: text} : m,
        ),
      }));
      await appendCachedMessage(sessionId, {...assistantMsg, content: text});
      await removePendingSend(userMsg.id);
      await triggerHaptic('success');
      sounds.receive();
      collectPair(content, text, sessionId);
      await saveSessionContext(sessionId, [
        ...state.messages,
        userMsg,
        {...assistantMsg, content: text},
      ]);
      await get().refreshSessions();
      set({streaming: false});
    };

    const handleError = async (err: any) => {
      if (err.name === 'AbortError') return;
      const msg = err instanceof ProviderError
        ? err.message
        : err.name === 'SSEHttpError'
          ? err.message
          : err.message || 'Request failed';
      toast.error(msg);
      set({error: msg, streaming: false});
      await triggerHaptic('error');
    };

    // ── Try local inference first ──────────────────────────────────────
    const hybridState = useHybridStore.getState();
    const route = hybridState.decideRoute(content);

    if (route.target === 'local') {
      try {
        const messages = [...state.messages, userMsg].map(m => ({
          role: m.role,
          content: m.content,
        }));
        let localAccumulated = '';
        const result = await hybridState.executeLocal(
          content,
          messages,
          (token: string) => {
            localAccumulated += token;
            const currentContent = localAccumulated;
            set(s => ({
              messages: s.messages.map(m =>
                m.id === assistantMsg.id
                  ? {...m, content: currentContent}
                  : m,
              ),
            }));
          },
        );

        if (result && result.text) {
          await finalizeSuccess(result.text);
          return;
        }
      } catch {}
      // Fall through to remote if local fails
    }

    // Offline-only mode: block remote fallback
    if (useHybridStore.getState().offlineOnly) {
      set({
        error: 'Offline mode: load a local engine in Settings',
        streaming: false,
      });
      toast.warn('Enable offline mode only when a local engine is loaded');
      await triggerHaptic('medium');
      return;
    }

    // ── Third-party provider ────────────────────────────────────────────
    if (route.target !== 'local' && route.target !== 'remote') {
      const providerId = route.target as string;
      const providers = useProvidersStore.getState().providers;
      const providerConfig = providers[providerId as keyof typeof providers];
      if (providerConfig?.apiKey) {
        try {
          const allMessages = [...state.messages, userMsg];
          const chatMessages = allMessages.map(m => ({
            role: m.role as 'user' | 'assistant' | 'system',
            content: m.content,
          }));
          const settings = useSettingsStore.getState();

          const result = await streamProviderChat(providerConfig, chatMessages, {
            maxTokens: settings.maxTokens,
            temperature: settings.temperature,
            topP: settings.topP,
            onToken: (token: string) => {
              accumulated += token;
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantMsg.id ? {...m, content: accumulated} : m,
                ),
              }));
            },
            signal: controller.signal,
          });

          await finalizeSuccess(result.text);
          return;
        } catch (err: any) {
          await handleError(err);
          return;
        }
      }
      // Provider not configured — fall through to self-hosted remote
    }

    // ── Self-hosted remote (SSE streaming) ──────────────────────────────
    const settings = useSettingsStore.getState();
    const allMessages = [...state.messages, userMsg];
    const allImages = allMessages.flatMap(m => m.images || []);
    const body = {
      messages: allMessages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      images: allImages.length > 0 ? allImages : undefined,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      top_p: settings.topP,
      top_k: settings.topK,
      repetition_penalty: settings.repetitionPenalty,
      session_id: sessionId,
    };

    let retries = 0;
    const maxRetries = 2;

    const attemptStream = async (): Promise<boolean> => {
      try {
        accumulated = '';
        set(s => ({
          messages: s.messages.map(m =>
            m.id === assistantMsg.id ? {...m, content: ''} : m,
          ),
        }));
        for await (const event of streamSSE(
          '/chat/stream',
          body,
          controller.signal,
        )) {
          // Token received
          if (event.data?.token) {
            accumulated += event.data.token as string;
            set(s => ({
              messages: s.messages.map(m =>
                m.id === assistantMsg.id
                  ? {...m, content: accumulated}
                  : m,
              ),
            }));
          }

          // Error from server
          if (event.status === 'error') {
            const errorMsg =
              (event.data?.error as string) ||
              event.message ||
              'Stream error';
            if (retries < maxRetries) {
              retries++;
              await new Promise(r => setTimeout(r, 500 * retries));
              return false;
            }
            set({error: errorMsg, streaming: false});
            toast.error(errorMsg);
            sounds.error();
            await triggerHaptic('error');
            return true;
          }

          // Stream complete
          if (event.status === 'complete') return true;
        }
        return true;
      } catch (err: any) {
        if (err.name === 'AbortError') return true;

        if (err.name === 'SSEHttpError') {
          toast.error(err.message);
          set({error: err.message, streaming: false});
          await triggerHaptic('error');
          return true;
        }

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
        set({
          error: 'Offline — message queued for retry',
          streaming: false,
        });
        await triggerHaptic('error');
        return true;
      }
    };

    try {
      let done = false;
      while (!done) {
        done = await attemptStream();
      }

      if (accumulated) {
        await finalizeSuccess(accumulated);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      set({error: err.message});
      await triggerHaptic('error');
    } finally {
      set({streaming: false});
      _abortController = null;
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

    // ── Shared state for all remote paths ───────────────────────────────
    const controller = new AbortController();
    _abortController = controller;
    let accumulated = '';

    const finalizeSuccess = async (text: string) => {
      set(s => ({
        messages: s.messages.map(m =>
          m.id === assistantMsg.id ? {...m, content: text} : m,
        ),
      }));
      await appendCachedMessage(sessionId, {...assistantMsg, content: text});
      await triggerHaptic('success');
      sounds.receive();
      const lastUserMsg = contextMessages[contextMessages.length - 1];
      if (lastUserMsg) {
        collectPair(lastUserMsg.content, text, sessionId);
      }
      await saveSessionContext(sessionId, [
        ...contextMessages,
        {...assistantMsg, content: text},
      ]);
      await get().refreshSessions();
      set({streaming: false});
    };

    const handleError = async (err: any) => {
      if (err.name === 'AbortError') return;
      const msg = err instanceof ProviderError
        ? err.message
        : err.name === 'SSEHttpError'
          ? err.message
          : err.message || 'Regeneration failed';
      toast.error(msg);
      sounds.error();
      set({error: msg, streaming: false});
      await triggerHaptic('error');
    };

    // ── Try local inference first ──────────────────────────────────────
    const hybridState = useHybridStore.getState();
    const route = hybridState.decideRoute(
      contextMessages[contextMessages.length - 1]?.content ?? '',
    );

    if (route.target === 'local') {
      try {
        let localAccumulated = '';
        const result = await hybridState.executeLocal(
          contextMessages[contextMessages.length - 1]?.content ?? '',
          contextMessages,
          (token: string) => {
            localAccumulated += token;
            const currentContent = localAccumulated;
            set(s => ({
              messages: s.messages.map(m =>
                m.id === assistantMsg.id
                  ? {...m, content: currentContent}
                  : m,
              ),
            }));
          },
        );
        if (result?.text) {
          await finalizeSuccess(result.text);
          return;
        }
      } catch {
        // Fall through to remote
      }
    }

    // Offline-only mode: block remote fallback
    if (useHybridStore.getState().offlineOnly) {
      set({
        error: 'Offline mode: load a local engine in Settings',
        streaming: false,
      });
      toast.warn('Enable offline mode only when a local engine is loaded');
      await triggerHaptic('medium');
      return;
    }

    // ── Third-party provider ────────────────────────────────────────────
    if (route.target !== 'local' && route.target !== 'remote') {
      const providerId = route.target as string;
      const providers = useProvidersStore.getState().providers;
      const providerConfig = providers[providerId as keyof typeof providers];
      if (providerConfig?.apiKey) {
        try {
          const chatMessages = contextMessages.map(m => ({
            role: m.role as 'user' | 'assistant' | 'system',
            content: m.content,
          }));
          const settings = useSettingsStore.getState();

          const result = await streamProviderChat(providerConfig, chatMessages, {
            maxTokens: settings.maxTokens,
            temperature: settings.temperature,
            topP: settings.topP,
            onToken: (token: string) => {
              accumulated += token;
              set(s => ({
                messages: s.messages.map(m =>
                  m.id === assistantMsg.id ? {...m, content: accumulated} : m,
                ),
              }));
            },
            signal: controller.signal,
          });

          await finalizeSuccess(result.text);
          return;
        } catch (err: any) {
          await handleError(err);
          return;
        }
      }
      // Provider not configured — fall through to self-hosted remote
    }

    // ── Self-hosted remote (SSE streaming) ──────────────────────────────
    try {
      for await (const event of streamSSE(
        `/session/${sessionId}/regenerate`,
        {messages: contextMessages},
        controller.signal,
      )) {
        if (event.data?.token) {
          accumulated += event.data.token as string;
          set(s => ({
            messages: s.messages.map(m =>
              m.id === assistantMsg.id
                ? {...m, content: accumulated}
                : m,
            ),
          }));
        }
        if (event.status === 'error') {
          const errorMsg =
            (event.data?.error as string) ||
            event.message ||
            'Regeneration failed';
          toast.error(errorMsg);
          sounds.error();
          set({error: errorMsg, streaming: false});
          await triggerHaptic('error');
          return;
        }
        if (event.status === 'complete') break;
      }
      if (accumulated) {
        await finalizeSuccess(accumulated);
      }
    } catch (err: any) {
      await handleError(err);
    } finally {
      set({streaming: false});
      _abortController = null;
    }
  },

  cancelStream: () => {
    _abortController?.abort();
    set({streaming: false});
  },

  recordFeedback: async (messageId: string, positive: boolean) => {
    try {
      await api.post('/feedback/workflow-record', {
        conversation_id: messageId,
        rating: positive ? 'thumbs_up' : 'thumbs_down',
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

    toast.info(
      `Retrying ${pending.length} queued message${pending.length > 1 ? 's' : ''}...`,
    );
    for (const send of pending) {
      set({activeSessionId: send.sessionId});
      await get().sendMessage(send.content);
    }
    set({offlineQueue: 0});
    toast.success('Messages synced');
  },
}));
