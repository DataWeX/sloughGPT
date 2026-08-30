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

const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
const MAX_RETRIES = 1;
const BASE_DELAY = 500;
const DEFAULT_TIMEOUT_MS = 30_000;

function parseErrorDetail(text: string, resStatus: number): string {
  try {
    const j = JSON.parse(text);
    const detail = j.detail ?? j.message ?? j.error;
    if (Array.isArray(detail)) {
      return detail
        .map((d: any) => (typeof d === 'string' ? d : d.msg ?? ''))
        .join('; ');
    }
    return detail || `Request failed (${resStatus})`;
  } catch {
    return text || `Request failed (${resStatus})`;
  }
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

  let retries = 0;

  while (true) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), DEFAULT_TIMEOUT_MS);

    try {
      const config: RequestInit = {method, headers, signal: ac.signal};
      if (body && method !== 'GET') {
        config.body = JSON.stringify(body);
      }

      const res = await fetch(`${baseUrl}${path}`, config);
      clearTimeout(timer);

      if (!res.ok) {
        const status = res.status;
        const isRetryable = RETRYABLE_STATUSES.has(status);

        if (isRetryable && retries < MAX_RETRIES) {
          retries++;
          const retryAfter = Number(res.headers?.get?.('Retry-After')) || 0;
          const delay =
            retryAfter > 0
              ? retryAfter * 1000
              : BASE_DELAY * Math.pow(2, retries - 1);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }

        const text = await res.text();
        throw new ApiError(status, parseErrorDetail(text, status), text);
      }

      const text = await res.text();
      if (!text) return undefined as T;
      const raw = JSON.parse(text);
      return unwrap<T>(raw);
    } catch (err) {
      clearTimeout(timer);

      if (err instanceof ApiError) throw err;

      // Caller-initiated abort is terminal — never retry
      if (ac.signal.aborted && err instanceof Error && err.name === 'AbortError') {
        throw new ApiError(0, 'Request timed out');
      }

      const isConnRefused =
        err instanceof Error &&
        (err.message === 'Failed to fetch' ||
          (err as any).cause?.code === 'ECONNREFUSED');

      if (retries < MAX_RETRIES) {
        retries++;
        await new Promise(r =>
          setTimeout(r, BASE_DELAY * Math.pow(2, retries - 1)),
        );
        continue;
      }

      const message = isConnRefused
        ? 'Connection unavailable — server may be starting up'
        : err instanceof Error
          ? err.message
          : 'Request failed';

      throw new ApiError(0, message);
    }
  }
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
    api.get<SearchResult[]>(
      `/chat/sessions/search?q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`,
    ),

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
  pullWeights: async (checkpoint: string) => {
    const baseUrl = await getApiUrl();
    return fetch(
      `${baseUrl}/training/checkpoints/${encodeURIComponent(checkpoint)}/export-mobile`,
    ).then(r => r.json());
  },
};
