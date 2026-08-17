/**
 * Message bookmarks — save important messages for quick access.
 * Stored in AsyncStorage.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/bookmarks';

export interface Bookmark {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  sessionId: string;
  sessionTitle?: string;
  savedAt: number;
}

let cached: Bookmark[] | null = null;

async function load(): Promise<Bookmark[]> {
  if (cached) return cached;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cached = JSON.parse(raw);
      return cached!;
    }
  } catch {}
  cached = [];
  return cached;
}

async function save(bookmarks: Bookmark[]): Promise<void> {
  cached = bookmarks;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
}

export async function getBookmarks(): Promise<Bookmark[]> {
  return load();
}

export async function addBookmark(
  content: string,
  role: 'user' | 'assistant',
  messageId: string,
): Promise<Bookmark> {
  const all = await load();
  // Dedup by content + message id
  const exists = all.find(b => b.content === content && b.sessionId === messageId);
  if (exists) return exists;

  const bookmark: Bookmark = {
    id: `bm-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    content,
    role,
    sessionId: messageId,
    savedAt: Date.now(),
  };
  all.unshift(bookmark);
  await save(all);
  return bookmark;
}

export async function removeBookmark(id: string): Promise<void> {
  const all = await load();
  await save(all.filter(b => b.id !== id));
}

export async function isBookmarked(content: string, messageId: string): Promise<boolean> {
  const all = await load();
  return all.some(b => b.content === content && b.sessionId === messageId);
}

export function _resetCache(): void {
  cached = null;
}
