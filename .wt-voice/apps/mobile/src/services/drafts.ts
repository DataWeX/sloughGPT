/**
 * Draft messages — auto-save unsent input per session.
 * Stored in AsyncStorage keyed by session ID.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/drafts';

let cached: Record<string, string> | null = null;

async function load(): Promise<Record<string, string>> {
  if (cached) return cached;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (raw) {
      cached = JSON.parse(raw);
      return cached!;
    }
  } catch {}
  cached = {};
  return cached;
}

async function save(drafts: Record<string, string>): Promise<void> {
  cached = drafts;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(drafts));
}

export async function getDraft(sessionId: string): Promise<string> {
  const all = await load();
  return all[sessionId] || '';
}

export async function saveDraft(sessionId: string, text: string): Promise<void> {
  const all = await load();
  if (text.trim()) {
    all[sessionId] = text;
  } else {
    delete all[sessionId];
  }
  await save(all);
}

export async function clearDraft(sessionId: string): Promise<void> {
  const all = await load();
  delete all[sessionId];
  await save(all);
}
