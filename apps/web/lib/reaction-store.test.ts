import { describe, expect, it, vi, beforeEach } from 'vitest'

const chatDBMock = {
  getKV: vi.fn().mockResolvedValue(undefined),
  setKV: vi.fn().mockResolvedValue(undefined),
  deleteKV: vi.fn().mockResolvedValue(undefined),
}

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

import { getReactions, addReaction, toggleReaction, removeReaction } from './reaction-store'

const KEY = 'sloughgpt-message-reactions'

describe('reaction-store', () => {
  beforeEach(() => {
    chatDBMock.getKV.mockClear()
    chatDBMock.setKV.mockClear()
    chatDBMock.deleteKV.mockClear()
  })

  it('returns an empty map for an unknown message', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('adds a reaction for a user and persists it', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    await addReaction('msg-1', '👍')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['user'] })
    expect(chatDBMock.setKV).toHaveBeenCalledWith(KEY, expect.objectContaining({
      value: { 'msg-1': { '👍': ['user'] } }
    }))
  })

  it('does not duplicate the same user for one emoji', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    await addReaction('msg-1', '👍')
    await addReaction('msg-1', '👍')
    await addReaction('msg-1', '👍', 'alice')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['user', 'alice'] })
  })

  it('removes a user from a reaction via toggle', async () => {
    chatDBMock.getKV
      .mockResolvedValueOnce({ value: { 'msg-1': { '👍': ['user', 'alice'] } } })
    await toggleReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['alice'] })
  })

  it('deletes the emoji key when the last user is removed via toggle', async () => {
    chatDBMock.getKV
      .mockResolvedValueOnce({ value: { 'msg-1': { '👍': ['user'] } } })
    await toggleReaction('msg-1', '👍')
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('is a no-op when toggling a reaction that does not exist', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({ '🔥': ['user'] })
  })

  it('toggles a reaction on for the user and off again', async () => {
    chatDBMock.getKV
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce({ value: { 'msg-1': { '🔥': ['user'] } } })
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({ '🔥': ['user'] })
    await toggleReaction('msg-1', '🔥')
    expect(await getReactions('msg-1')).toEqual({})
  })

  it('keeps separate reactions per message', async () => {
    chatDBMock.getKV.mockResolvedValueOnce(undefined)
    await addReaction('msg-a', '👍')
    await addReaction('msg-b', '🔥')
    expect(await getReactions('msg-a')).toEqual({ '👍': ['user'] })
    expect(await getReactions('msg-b')).toEqual({ '🔥': ['user'] })
  })

  it('removeReaction removes a user from an emoji and cleans empty entries', async () => {
    chatDBMock.getKV.mockResolvedValueOnce({ value: { 'msg-1': { '👍': ['user', 'alice'] } } })
    await removeReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({ '👍': ['alice'] })
  })

  it('removeReaction deletes the message entry when the last reaction is removed', async () => {
    chatDBMock.getKV.mockResolvedValueOnce({ value: { 'msg-1': { '👍': ['user'] } } })
    await removeReaction('msg-1', '👍', 'user')
    expect(await getReactions('msg-1')).toEqual({})
  })
})
