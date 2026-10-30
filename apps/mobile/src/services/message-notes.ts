/**
 * Message notes — personal annotations attached to a single message.
 * Stored in AsyncStorage, keyed by `${sessionId}:${messageId}` so notes are
 * namespaced per conversation (branch/clone get their own namespace).
 * Mobile twin of web's IndexedDB `chatDB.getMessageNotes` family.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/message-notes';

export interface MessageNote {
  id: string;
  sessionId: string;
  messageId: string;
  content: string;
  createdAt: number;
  updatedAt: number;
}

let cached: MessageNote[] | null = null;

async function load(): Promise<MessageNote[]> {
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

async function save(notes: MessageNote[]): Promise<void> {
  cached = notes;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
}

export async function getMessageNotes(sessionId: string): Promise<MessageNote[]> {
  const all = await load();
  return all.filter(n => n.sessionId === sessionId);
}

export async function saveMessageNote(note: MessageNote): Promise<void> {
  const all = await load();
  const idx = all.findIndex(n => n.id === note.id);
  if (idx >= 0) {
    all[idx] = note;
  } else {
    all.push(note);
  }
  await save(all);
}

export async function removeMessageNote(
  sessionId: string,
  messageId: string,
): Promise<void> {
  const all = await load();
  await save(all.filter(n => !(n.sessionId === sessionId && n.messageId === messageId)));
}
