'use client'

const REACTIONS_KEY = 'sloughgpt-message-reactions'

export type ReactionMap = Record<string, Record<string, string[]>>

function loadReactions(): ReactionMap {
  try {
    return JSON.parse(localStorage.getItem(REACTIONS_KEY) || '{}')
  } catch {
    return {}
  }
}

function saveReactions(reactions: ReactionMap): void {
  localStorage.setItem(REACTIONS_KEY, JSON.stringify(reactions))
}

export function getReactions(messageId: string): Record<string, string[]> {
  const all = loadReactions()
  return all[messageId] || {}
}

export function addReaction(messageId: string, emoji: string, userId: string = 'user'): void {
  const all = loadReactions()
  if (!all[messageId]) all[messageId] = {}
  if (!all[messageId][emoji]) all[messageId][emoji] = []
  if (!all[messageId][emoji].includes(userId)) {
    all[messageId][emoji].push(userId)
  }
  saveReactions(all)
}

function removeReaction(messageId: string, emoji: string, userId: string = 'user'): void {
  const all = loadReactions()
  if (all[messageId]?.[emoji]) {
    all[messageId][emoji] = all[messageId][emoji].filter(u => u !== userId)
    if (all[messageId][emoji].length === 0) {
      delete all[messageId][emoji]
    }
    if (Object.keys(all[messageId]).length === 0) {
      delete all[messageId]
    }
  }
  saveReactions(all)
}

export function toggleReaction(messageId: string, emoji: string, userId: string = 'user'): void {
  const reactions = getReactions(messageId)
  if (reactions[emoji]?.includes(userId)) {
    removeReaction(messageId, emoji, userId)
  } else {
    addReaction(messageId, emoji, userId)
  }
}
