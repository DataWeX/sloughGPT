import AsyncStorage from '@react-native-async-storage/async-storage';
import type {SearchResult} from '../types';

const API_URL_KEY = '@sloughgpt/api_url';
const DEFAULT_URL = 'http://localhost:8000';

export async function getApiUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(API_URL_KEY);
  return stored || DEFAULT_URL;
}

export async function setApiUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(API_URL_KEY, url);
}

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(status: number, message: string, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/** Standard response envelope from backend. */
interface StandardResponse<T> {
  status: 'success' | 'error';
  data: T;
  message?: string;
  meta?: Record<string, unknown>;
}

/** Unwrap StandardResponse — extracts `data` field if present. */
function unwrap<T>(raw: unknown): T {
  if (raw && typeof raw === 'object' && 'status' in raw && 'data' in raw) {
    return (raw as StandardResponse<T>).data;
  }
  return raw as T;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const baseUrl = await getApiUrl();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const config: RequestInit = {method, headers};
  if (body && method !== 'GET') {
    config.body = JSON.stringify(body);
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(`${baseUrl}${path}`, config);
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        const msg =
          (data as any)?.detail || (data as any)?.error || res.statusText;
        throw new ApiError(res.status, msg, data);
      }
      const text = await res.text();
      if (!text) return undefined as T;
      const raw = JSON.parse(text);
      return unwrap<T>(raw);
    } catch (err) {
      if (err instanceof ApiError && err.status < 500) throw err;
      lastError = err as Error;
      if (attempt === 0) {
        await new Promise(r => setTimeout(r, 500));
      }
    }
  }
  throw lastError;
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, body),
  delete: <T>(path: string) => request<T>('DELETE', path),

  /** Upload a file (image/audio) via multipart form data. */
  upload: async <T>(path: string, formData: FormData): Promise<T> => {
    const baseUrl = await getApiUrl();
    const res = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => null);
      throw new ApiError(res.status, (data as any)?.detail || res.statusText, data);
    }
    const text = await res.text();
    if (!text) return undefined as T;
    const raw = JSON.parse(text);
    return unwrap<T>(raw);
  },

  /** Sync offline messages with the server. */
  sync: <T>(body: {pending_messages: any[]; last_sync_timestamp?: number}) =>
    request<T>('POST', '/mobile/sync', body),

  /** Rename a session. */
  renameSession: (sessionId: string, name: string) =>
    api.put(`/chat/sessions/${sessionId}`, {name}),

  /** Archive or unarchive a session. */
  archiveSession: (sessionId: string, archived: boolean) =>
    api.put(`/chat/sessions/${sessionId}`, {archived}),

  /** Search across all sessions. */
  searchSessions: (q: string, limit?: number) =>
    api.get<SearchResult[]>(`/chat/sessions/search?q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`),

  /** Send a voice message (audio upload) to a session. */
  sendVoiceMessage: (sessionId: string, audioUri: string, durationMs: number) => {
    const formData = new FormData();
    formData.append('file', {
      uri: audioUri,
      type: 'audio/m4a',
      name: 'voice.m4a',
    } as any);
    formData.append('duration_ms', String(durationMs));
    return api.upload<{message_id: string; audio_path: string; session_id: string}>(
      `/chat/voice/${sessionId}`,
      formData,
    );
  },

  /** Check server connectivity. */
  syncStatus: <T>() => request<T>('GET', '/mobile/sync/status'),

  /** Trigger on-device training on server with conversation pairs. */
  mobileTrain: <T>(body: {pairs: any[]; checkpoint: string}) =>
    request<T>('POST', '/mobile/train', body),

  /** Train from server-side inference logs (no mobile data needed). */
  trainFromSessions: <T>(body?: {limit?: number; min_length?: number; model?: string}) =>
    request<T>('POST', '/mobile/train/from-sessions', body || {}),

  /** Get auto-trainer status. */
  getAutoTrainStatus: <T>() => request<T>('GET', '/mobile/train/auto-status'),

  /** Pull latest weights from a trained checkpoint. */
  pullWeights: (checkpoint: string) =>
    fetch(`${'' /* resolved at call site */}/auto-train/checkpoints/${encodeURIComponent(checkpoint)}/export-mobile`)
      .then(r => r.json()),
};
