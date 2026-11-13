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
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] cacheMessages failed:', e);
  }
}

export async function getCachedMessages(sessionId: string): Promise<Message[]> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    return all[sessionId] || [];
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] getCachedMessages failed:', e);
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
    // Cap at 200 messages per session
    if (msgs.length > 200) {
      all[sessionId] = msgs.slice(-200);
    }
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] appendCachedMessage failed:', e);
  }
}

export async function removeCachedMessage(sessionId: string, messageId: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    const msgs = all[sessionId] || [];
    all[sessionId] = msgs.filter(m => m.id !== messageId);
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] removeCachedMessage failed:', e);
  }
}

export async function clearCachedMessages(sessionId: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(MESSAGES_KEY);
    const all: Record<string, Message[]> = raw ? JSON.parse(raw) : {};
    delete all[sessionId];
    await AsyncStorage.setItem(MESSAGES_KEY, JSON.stringify(all));
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] clearCachedMessages failed:', e);
  }
}

// ── Pending Sends ───────────────────────────────────────────────────────────

export async function addPendingSend(send: PendingSend): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    pending.push(send);
    await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] addPendingSend failed:', e);
  }
}

export async function getPendingSends(): Promise<PendingSend[]> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] getPendingSends failed:', e);
    return [];
  }
}

export async function removePendingSend(id: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    await AsyncStorage.setItem(
      PENDING_KEY,
      JSON.stringify(pending.filter(p => p.id !== id)),
    );
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] removePendingSend failed:', e);
  }
}

export async function incrementPendingRetry(id: string): Promise<void> {
  try {
    const raw = await AsyncStorage.getItem(PENDING_KEY);
    const pending: PendingSend[] = raw ? JSON.parse(raw) : [];
    const send = pending.find(p => p.id === id);
    if (send) send.retryCount++;
    await AsyncStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] incrementPendingRetry failed:', e);
  }
}

export async function clearPendingSends(): Promise<void> {
  try {
    await AsyncStorage.removeItem(PENDING_KEY);
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] clearPendingSends failed:', e);
  }
}

// ── Active Session ──────────────────────────────────────────────────────────

export async function getActiveSessionId(): Promise<string | null> {
  try {
    return AsyncStorage.getItem(SESSION_KEY);
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] getActiveSessionId failed:', e);
    return null;
  }
}

export async function setActiveSessionId(sessionId: string | null): Promise<void> {
  try {
    if (sessionId) {
      await AsyncStorage.setItem(SESSION_KEY, sessionId);
    } else {
      await AsyncStorage.removeItem(SESSION_KEY);
    }
  } catch (e) {
    if (__DEV__) console.warn('[offline-cache] setActiveSessionId failed:', e);
  }
}

// Aliases for backward compatibility
export const cacheActiveSessionId = setActiveSessionId;
export const getCachedActiveSessionId = getActiveSessionId;
