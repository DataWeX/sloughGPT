import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/starred-sessions';

let cache: string[] | null = null;

async function load(): Promise<string[]> {
  if (cache) return cache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cache = JSON.parse(raw);
      return cache!;
    }
  } catch {}
  cache = [];
  return cache;
}

async function save(ids: string[]): Promise<void> {
  cache = ids;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
}

/** Get all starred session IDs. */
export async function getStarredIds(): Promise<string[]> {
  return load();
}

/** Star a session. */
export async function starSession(sessionId: string): Promise<void> {
  const ids = await load();
  if (!ids.includes(sessionId)) {
    ids.unshift(sessionId);
    await save(ids);
  }
}

/** Unstar a session. */
export async function unstarSession(sessionId: string): Promise<void> {
  const ids = await load();
  const filtered = ids.filter(id => id !== sessionId);
  if (filtered.length !== ids.length) {
    await save(filtered);
  }
}

/** Check if a session is starred. */
export async function isStarred(sessionId: string): Promise<boolean> {
  const ids = await load();
  return ids.includes(sessionId);
}

/** Reset the in-memory cache. Used in tests. */
export function _resetCache(): void {
  cache = null;
}
