import { describe, it, expect, beforeEach } from 'vitest'
import { getReactions, addReaction, toggleReaction } from './reaction-store'

const KEY = 'sloughgpt-message-reactions'

describe('reaction-store', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('stores a reaction for a message', () => {
    addReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({ '👍': 1 })
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual({ 'msg-1': { '👍': 1 } })
  })

  it('increments reaction count for the same emoji', () => {
    addReaction('msg-1', '👍')
    addReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({ '👍': 2 })
  })

  it('removes a reaction via toggle', () => {
    addReaction('msg-1', '👍')
    addReaction('msg-1', '👍')
    toggleReaction('msg-1', '👍')
    expect(getReactions('msg-1')).toEqual({ '👍': 1 })
  })

  it('deletes the emoji key when the last reaction is removed via toggle', () => {
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

  it('toggles a reaction on for the user and off again', () => {
    toggleReaction('msg-1', '🔥')
    expect(getReactions('msg-1')).toEqual({ '🔥': 1 })
    toggleReaction('msg-1', '🔥')
    expect(getReactions('msg-1')).toEqual({})
  })

  it('keeps separate reactions per message', () => {
    addReaction('msg-a', '👍')
    addReaction('msg-b', '🔥')
    expect(getReactions('msg-a')).toEqual({ '👍': 1 })
    expect(getReactions('msg-b')).toEqual({ '🔥': 1 })
  })
})
