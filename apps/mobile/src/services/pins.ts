import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/pinned-messages';

let cache: Record<string, string[]> | null = null;

async function load(): Promise<Record<string, string[]>> {
  if (cache) return cache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cache = JSON.parse(raw);
      return cache!;
    }
  } catch {}
  cache = {};
  return cache;
}

async function save(data: Record<string, string[]>): Promise<void> {
  cache = data;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

/** Get pinned message IDs for a session. */
export async function getPinnedIds(sessionId: string): Promise<string[]> {
  const data = await load();
  return data[sessionId] || [];
}

/** Pin a message in a session. */
export async function pinMessage(sessionId: string, messageId: string): Promise<void> {
  const data = await load();
  if (!data[sessionId]) data[sessionId] = [];
  if (!data[sessionId].includes(messageId)) {
    data[sessionId].unshift(messageId);
    await save(data);
  }
}

/** Unpin a message from a session. */
export async function unpinMessage(sessionId: string, messageId: string): Promise<void> {
  const data = await load();
  if (data[sessionId]) {
    data[sessionId] = data[sessionId].filter(id => id !== messageId);
    if (data[sessionId].length === 0) delete data[sessionId];
    await save(data);
  }
}

/** Check if a message is pinned. */
export async function isPinned(sessionId: string, messageId: string): Promise<boolean> {
  const ids = await getPinnedIds(sessionId);
  return ids.includes(messageId);
}

/** Reset the in-memory cache. Used in tests. */
export function _resetCache(): void {
  cache = null;
}
