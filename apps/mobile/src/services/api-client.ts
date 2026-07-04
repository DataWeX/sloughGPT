import AsyncStorage from '@react-native-async-storage/async-storage';

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
      return JSON.parse(text) as T;
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
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    if (!res.ok) {
      const data = await res.json().catch(() => null);
      throw new ApiError(res.status, (data as any)?.detail || res.statusText, data);
    }
    const text = await res.text();
    if (!text) return undefined as T;
    return JSON.parse(text) as T;
  },

  /** Sync offline messages with the server. */
  sync: <T>(body: {pending_messages: any[]; last_sync_timestamp?: number}) =>
    request<T>('POST', '/mobile/sync', body),

  /** Check server connectivity. */
  syncStatus: <T>() => request<T>('GET', '/mobile/sync/status'),
};
