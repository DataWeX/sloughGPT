import { describe, expect, it, vi, beforeEach } from 'vitest'

const store = new Map<string, unknown>()

const { chatDBMock } = vi.hoisted(() => {
  const chatDBMock = {
    getKV: vi.fn(async <T>(key: string): Promise<T | undefined> => {
      const entry = store.get(key)
      return entry as T | undefined
    }),
    setKV: vi.fn(async <T>(key: string, value: T): Promise<void> => {
      store.set(key, value)
    }),
    deleteKV: vi.fn(async (key: string): Promise<void> => {
      store.delete(key)
    }),
  }
  return { chatDBMock }
})

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

import { getReactions, addReaction, toggleReaction, removeReaction } from './reaction-store'

const KEY = 'sloughgpt-message-reactions'

describe('reaction-store', () => {
  beforeEach(() => {
    store.clear()
    chatDBMock.getKV.mockClear()
    chatDBMock.setKV.mockClear()
    chatDBMock.deleteKV.mockClear()
  })

  it('returns an empty map for an unknown message', async () => {
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('adds a reaction for a user and persists it', async () => {
    await addReaction('msg-1', '👍')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['user'] })
    expect(chatDBMock.setKV).toHaveBeenCalledWith(KEY, { value: { 'msg-1': { '👍': ['user'] } } })
  })

  it('does not duplicate the same user for one emoji', async () => {
    await addReaction('msg-1', '👍')
    await addReaction('msg-1', '👍')
    await addReaction('msg-1', '👍', 'alice')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['user', 'alice'] })
  })

  it('removes a user from a reaction via toggle', async () => {
    await addReaction('msg-1', '👍')
    await addReaction('msg-1', '👍', 'alice')
    await toggleReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['alice'] })
  })

  it('deletes the emoji key when the last user is removed via toggle', async () => {
    await addReaction('msg-1', '👍')
    await toggleReaction('msg-1', '👍')
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('is a no-op when toggling a reaction that does not exist', async () => {
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({ '🔥': ['user'] })
  })

  it('toggles a reaction on for the user and off again', async () => {
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({ '🔥': ['user'] })
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('keeps separate reactions per message', async () => {
    await addReaction('msg-a', '👍')
    await addReaction('msg-b', '🔥')
    expect(await getReactions('msg-a')).toEqual({ '👍': ['user'] })
    expect(await getReactions('msg-b')).toEqual({ '🔥': ['user'] })
  })

  it('removeReaction removes a user from an emoji and cleans empty entries', async () => {
    store.set(KEY, { value: { 'msg-1': { '👍': ['user', 'alice'] } } })
    await removeReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['alice'] })
  })

  it('removeReaction deletes the message entry when the last reaction is removed', async () => {
    store.set(KEY, { value: { 'msg-1': { '👍': ['user'] } } })
    await removeReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({})
  })
})
