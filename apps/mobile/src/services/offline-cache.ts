/**
 * Offline message cache.
 * Stores messages and pending sends in AsyncStorage for offline resilience.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import type {Message} from '../types';

const MESSAGES_KEY = '@sloughgpt/messages';
const PENDING_KEY = '@sloughgpt/pending_sends';
const SESSION_KEY = '@sloughgpt/active_session';

export interface PendingSend {
  id: string;
  sessionId: string;
  content: string;
  timestamp: number;
  retryCount: number;
}

// ── Messages ────────────────────────────────────────────────────────────────

export async function cacheMessages(sessionId: string, messages: Message[]): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    all[sessionId] = messages;
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch {
    // best-effort
  }
}

export async function getCachedMessages(sessionId: string): Promise<Message[]> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    return all[sessionId] || [];
  } catch {
    return [];
  }
}

export async function appendCachedMessage(sessionId: string, message: Message): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    const msgs = all[sessionId] || [];
    msgs.push(message);
    all[sessionId] = msgs;
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch {
    // best-effort
  }
}

export async function removeCachedMessage(sessionId: string, messageId: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    const msgs = all[sessionId] || [];
    all[sessionId] = msgs.filter(m => m.id !== messageId);
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch {
    // best-effort
  }
}

// ── Pending Sends ───────────────────────────────────────────────────────────

export async function addPendingSend(send: PendingSend): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    pending.push(send);
    await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch {
    // best-effort
  }
}

export async function getPendingSends(): Promise<PendingSend[]> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function removePendingSend(id: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    await AsyncStorage.setItem(
      PENDING_KEY,
      JSON.stringify(pending.filter(s => s.id !== id)),
    );
  } catch {
    // best-effort
  }
}

export async function incrementPendingRetry(id: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    const item = pending.find(s => s.id === id);
    if (item) item.retryCount += 1;
    await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch {
    // best-effort
  }
}

// ── Active Session ──────────────────────────────────────────────────────────

export async function cacheActiveSessionId(sessionId: string | null): Promise<void> {
  try {
    if (sessionId) {
      await AsyncStorage.setItem(SESSION_KEY, sessionId);
    } else {
      await AsyncStorage.removeItem(SESSION_KEY);
    }
  } catch {
    // best-effort
  }
}

export async function getCachedActiveSessionId(): Promise<string | null> {
  try {
    return await AsyncStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}
