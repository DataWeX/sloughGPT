'use client'

const REACTIONS_KEY = 'sloughgpt-message-reactions'

export type ReactionMap = Record<string, Record<string, number>>

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

export function getReactions(messageId: string): Record<string, number> {
  const all = loadReactions()
  return all[messageId] || {}
}

export function addReaction(messageId: string, emoji: string): void {
  const all = loadReactions()
  if (!all[messageId]) all[messageId] = {}
  all[messageId][emoji] = (all[messageId][emoji] || 0) + 1
  saveReactions(all)
}

function removeReaction(messageId: string, emoji: string): void {
  const all = loadReactions()
  if (all[messageId]?.[emoji]) {
    all[messageId][emoji] = Math.max(0, (all[messageId][emoji] || 0) - 1)
    if (all[messageId][emoji] === 0) {
      delete all[messageId][emoji]
    }
    if (Object.keys(all[messageId]).length === 0) {
      delete all[messageId]
    }
  }
  saveReactions(all)
}

export function toggleReaction(messageId: string, emoji: string): void {
  const reactions = getReactions(messageId)
  if (reactions[emoji] && reactions[emoji] > 0) {
    removeReaction(messageId, emoji)
  } else {
    addReaction(messageId, emoji)
  }
}
