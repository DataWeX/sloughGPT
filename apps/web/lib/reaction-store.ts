'use client'

import { chatDB } from '@/lib/db'

export type ReactionMap = Record<string, Record<string, string[]>>

const REACTIONS_KEY = 'sloughgpt-message-reactions'

async function loadReactions(): Promise<ReactionMap> {
  try {
    const entry = await chatDB.getKV<{ value: ReactionMap }>(REACTIONS_KEY)
    return entry?.value ?? {}
  } catch {
    return {}
  }
}

async function saveReactions(reactions: ReactionMap): Promise<void> {
  await chatDB.setKV(REACTIONS_KEY, { value: reactions })
}

export async function getReactions(messageId: string): Promise<Record<string, string[]>> {
  const all = await loadReactions()
  return all[messageId] || {}
}

export async function addReaction(messageId: string, emoji: string, userId: string = 'user'): Promise<void> {
  const all = await loadReactions()
  if (!all[messageId]) all[messageId] = {}
  if (!all[messageId][emoji]) all[messageId][emoji] = []
  if (!all[messageId][emoji].includes(userId)) {
    all[messageId][emoji].push(userId)
  }
  await saveReactions(all)
}

export async function removeReaction(messageId: string, emoji: string, userId: string = 'user'): Promise<void> {
  const all = await loadReactions()
  if (all[messageId]?.[emoji]) {
    all[messageId][emoji] = all[messageId][emoji].filter(u => u !== userId)
    if (all[messageId][emoji].length === 0) {
      delete all[messageId][emoji]
    }
    if (Object.keys(all[messageId]).length === 0) {
      delete all[messageId]
    }
  }
  await saveReactions(all)
}

export async function toggleReaction(messageId: string, emoji: string, userId: string = 'user'): Promise<void> {
  const reactions = await getReactions(messageId)
  if (reactions[emoji]?.includes(userId)) {
    await removeReaction(messageId, emoji, userId)
  } else {
    await addReaction(messageId, emoji, userId)
  }
}
