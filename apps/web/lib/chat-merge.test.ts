// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mergeSessionMessages, loadMergedMessages } from './chat-merge'
import { sessionController } from '@/lib/session-controller'
import type { ChatMessage } from '@/lib/chat-utils'

vi.mock('@/lib/session-controller', () => ({
  sessionController: {
    fetchMessages: vi.fn(async () => []),
  },
}))

const fetchMessages = vi.mocked(sessionController.fetchMessages)

function msg(role: 'user' | 'assistant', content: string, id?: string): ChatMessage {
  return { id: id || `${role}-${content}`, role, content, timestamp: new Date('2026-01-01') }
}

beforeEach(() => {
  fetchMessages.mockReset()
  fetchMessages.mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
})

describe('mergeSessionMessages — chronology (card 20261116_124020)', () => {
  // The legacy merge did [...remote, ...localOnly]: an early local-only
  // message (its context save failed, so the backend never saw it) landed
  // AFTER the newest backend reply. This is the ordering defect itself.
  it('keeps an early unsynced local message in position, not appended at the end', () => {
    const local = [
      msg('user', 'first'),
      msg('assistant', 'reply-1'),
      msg('user', 'second'),
      msg('assistant', 'reply-2'),
    ]
    const remote = [
      { role: 'assistant', content: 'reply-1' },
      { role: 'user', content: 'second' },
      { role: 'assistant', content: 'reply-2' },
    ]

    const merged = mergeSessionMessages(local, remote)

    expect(merged.map((m) => m.content)).toEqual(['first', 'reply-1', 'second', 'reply-2'])
  })

  it('inserts a backend-only middle message at its position, not only in the tail', () => {
    const local = [msg('user', 'q1'), msg('assistant', 'a1'), msg('user', 'q2')]
    const remote = [
      { role: 'user', content: 'q1' },
      { role: 'assistant', content: 'a1' },
      { role: 'user', content: 'q1.5' }, // said in another surface, local copy missing
      { role: 'user', content: 'q2' },
    ]

    const merged = mergeSessionMessages(local, remote)

    expect(merged.map((m) => m.content)).toEqual(['q1', 'a1', 'q1.5', 'q2'])
  })

  it('keeps the LOCAL message identity (real timestamp/id) on matched pairs', () => {
    const localTs = new Date('2026-02-02T10:00:00Z')
    const local = [{ ...msg('user', 'hi', 'local-id'), timestamp: localTs }]
    const remote = [{ role: 'user', content: 'hi', id: 'backend-id' }]

    const merged = mergeSessionMessages(local, remote)

    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe('local-id')
    expect(merged[0].timestamp).toBe(localTs)
  })

  it('does not collapse repeated identical texts (dedup is position-aware)', () => {
    const local = [msg('user', 'ok'), msg('user', 'ok'), msg('assistant', 'done')]
    const remote = [
      { role: 'user', content: 'ok' },
      { role: 'assistant', content: 'done' },
    ]

    const merged = mergeSessionMessages(local, remote)

    expect(merged.map((m) => m.content)).toEqual(['ok', 'ok', 'done'])
  })

  it('returns local untouched when the backend has nothing', () => {
    const local = [msg('user', 'a'), msg('assistant', 'b')]
    expect(mergeSessionMessages(local, [])).toBe(local)
  })

  it('maps backend-only conversations to ChatMessage with stable ids', () => {
    const merged = mergeSessionMessages([], [{ role: 'user', content: 'hi' }])
    expect(merged).toHaveLength(1)
    expect(merged[0].id).toBe('remote_0')
    expect(merged[0].role).toBe('user')
  })
})

describe('loadMergedMessages — fetch seam', () => {
  it('merges the fetched backend copy with the local copy', async () => {
    fetchMessages.mockResolvedValue([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'reply' },
    ])
    const local = [msg('user', 'first'), msg('assistant', 'reply'), msg('user', 'unsaved-tail')]

    const merged = await loadMergedMessages('sess-1', local)

    expect(fetchMessages).toHaveBeenCalledWith('sess-1', expect.anything())
    expect(merged.map((m) => m.content)).toEqual(['first', 'reply', 'unsaved-tail'])
  })

  it('drops persisted empty assistant messages on both sides', async () => {
    fetchMessages.mockResolvedValue([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: '' },
    ])
    const local = [msg('user', 'hi'), msg('assistant', '')]

    const merged = await loadMergedMessages('sess-1', local)

    expect(merged.map((m) => m.content)).toEqual(['hi'])
  })

  it('degrades to the filtered local copy when the backend fetch fails', async () => {
    fetchMessages.mockRejectedValue(new Error('offline'))
    const local = [msg('user', 'kept'), msg('assistant', '')]

    const merged = await loadMergedMessages('sess-1', local)

    expect(merged.map((m) => m.content)).toEqual(['kept'])
  })
})
