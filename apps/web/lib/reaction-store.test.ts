import { describe, expect, it, beforeEach } from 'vitest'
import {
  getReactions,
  addReaction,
  toggleReaction,
} from './reaction-store'

const KEY = 'sloughgpt-message-reactions'

describe('reaction-store', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('returns an empty map for an unknown message', () => {
    expect(getReactions('msg-1')).toEqual({})
  })

  it('returns an empty map when localStorage holds invalid JSON', () => {
    localStorage.setItem(KEY, '{broken')
    expect(getReactions('msg-1')).toEqual({})
  })

  it('adds a reaction for a user and persists it', () => {
    addReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({ '👍': ['user'] })
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({ 'msg-1': { '👍': ['user'] } })
  })

  it('does not duplicate the same user for one emoji', () => {
    addReaction('msg-1', '👍')
    addReaction('msg-1', '👍')
    addReaction('msg-1', '👍', 'alice')
    expect(getReactions('msg-1')).toEqual({ '👍': ['user', 'alice'] })
  })

  it('removes a user from a reaction via toggle', () => {
    addReaction('msg-1', '👍')
    addReaction('msg-1', '👍', 'alice')
    toggleReaction('msg-1', '👍', 'user')
    expect(getReactions('msg-1')).toEqual({ '👍': ['alice'] })
  })

  it('deletes the emoji key when the last user is removed via toggle', () => {
    addReaction('msg-1', '👍')
    toggleReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({})
  })

  it('deletes the message key when its last reaction is removed via toggle', () => {
    addReaction('msg-1', '👍')
    toggleReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({})
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({})
  })

  it('is a no-op when toggling a reaction that does not exist', () => {
    toggleReaction('msg-1', '🔥')
    expect(getReactions('msg-1')).toEqual({ '🔥': ['user'] })
  })

  it('toggles a reaction on for the user and off again', () => {
    toggleReaction('msg-1', '🔥')
    expect(getReactions('msg-1')).toEqual({ '🔥': ['user'] })
    toggleReaction('msg-1', '🔥')
    expect(getReactions('msg-1')).toEqual({})
  })

  it('keeps separate reactions per message', () => {
    addReaction('msg-a', '👍')
    addReaction('msg-b', '🔥')
    expect(getReactions('msg-a')).toEqual({ '👍': ['user'] })
    expect(getReactions('msg-b')).toEqual({ '🔥': ['user'] })
  })
})
