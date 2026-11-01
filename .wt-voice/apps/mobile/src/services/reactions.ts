/**
 * Message reactions — emoji reactions on messages.
 * Stored in AsyncStorage keyed by message ID.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@sloughgpt/reactions';

export const REACTION_EMOJIS = ['❤️', '🔥', '😂', '😮', '👍', '👎', '🎉', '🤔', '👀', '💯'] as const;

export type ReactionEmoji = typeof REACTION_EMOJIS[number];

export interface MessageReactions {
  [messageId: string]: ReactionEmoji[];
}

let cached: MessageReactions | null = null;

async function load(): Promise<MessageReactions> {
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

async function save(reactions: MessageReactions): Promise<void> {
  cached = reactions;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(reactions));
}

export async function getMessageReactions(messageId: string): Promise<ReactionEmoji[]> {
  const all = await load();
  return all[messageId] || [];
}

export async function toggleReaction(messageId: string, emoji: ReactionEmoji): Promise<ReactionEmoji[]> {
  const all = await load();
  const current = all[messageId] || [];

  let updated: ReactionEmoji[];
  if (current.includes(emoji)) {
    updated = current.filter(e => e !== emoji);
  } else {
    updated = [...current, emoji];
  }

  all[messageId] = updated;
  await save(all);
  return updated;
}

export async function getAllReactions(): Promise<MessageReactions> {
  return load();
}
