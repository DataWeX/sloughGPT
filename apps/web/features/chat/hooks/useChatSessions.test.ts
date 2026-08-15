import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

const {
  mockSaveSession, mockLoadSession, mockLoadSessions,
  mockDeleteSession, mockUpdateSessionFn,
  mockCreate, mockUpdate, mockDelete, mockFetchMessages,
  mockAddGlobalError, mockGetKV, mockSetKV, mockDeleteKV, mockGetDraft,
} = vi.hoisted(() => {
  const kvStore = new Map<string, unknown>()
  return {
    mockSaveSession: vi.fn(),
    mockLoadSession: vi.fn(),
    mockLoadSessions: vi.fn(),
    mockDeleteSession: vi.fn(),
    mockUpdateSessionFn: vi.fn(),
    mockCreate: vi.fn(),
    mockUpdate: vi.fn(),
    mockDelete: vi.fn(),
    mockFetchMessages: vi.fn(),
    mockAddGlobalError: vi.fn(),
    mockGetKV: vi.fn(async (key: string) => kvStore.get(key)),
    mockSetKV: vi.fn(async (key: string, value: unknown) => { kvStore.set(key, value) }),
    mockDeleteKV: vi.fn(async (key: string) => { kvStore.delete(key) }),
    mockGetDraft: vi.fn(async () => ''),
  }
})

vi.mock('@/lib/db', () => ({
  chatDB: {
    saveSession: mockSaveSession,
    loadSession: mockLoadSession,
    loadSessions: mockLoadSessions,
    deleteSession: mockDeleteSession,
    updateSession: mockUpdateSessionFn,
    getKV: mockGetKV,
    setKV: mockSetKV,
    deleteKV: mockDeleteKV,
    getDraft: mockGetDraft,
  },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: {
    create: mockCreate,
    update: mockUpdate,
    delete: mockDelete,
    fetchMessages: mockFetchMessages,
  },
}))

vi.mock('@/lib/error-store', () => ({
  addGlobalError: mockAddGlobalError,
}))

vi.mock('@/lib/chat-utils', () => ({
  CURRENT_SESSION_KEY: 'man_current_conversation',
  generateSessionId: () => 'session_test',
}))

import { useChatSessions } from './useChatSessions'

describe('useChatSessions', () => {
  const defaultOpts = () => ({
    setMessages: vi.fn(),
    setInput: vi.fn(),
    setSessionSaved: vi.fn(),
    setSessionLoading: vi.fn(),
    sessionIdRef: { current: 's1' } as React.MutableRefObject<string>,
    showToast: vi.fn(),
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockLoadSessions.mockResolvedValue([])
    mockCreate.mockResolvedValue(undefined)
    mockUpdate.mockResolvedValue(undefined)
    mockDelete.mockResolvedValue(undefined)
  })

  it('returns session state with defaults', () => {
    const { result } = renderHook(() => useChatSessions(defaultOpts()))
    expect(result.current.sessions).toEqual([])
    expect(result.current.sidebarConversations).toEqual([])
  })

  it('saveSessionToStorage saves to DB and creates remote session', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    const msgs = [{ id: '1', role: 'user' as const, content: 'What is machine learning', timestamp: new Date() }]
    await act(async () => { await result.current.saveSessionToStorage(msgs, 's1') })
    const saved = mockSaveSession.mock.calls[0][0]
    expect(saved.id).toBe('s1')
    expect(saved.name).toContain('machine learning')
    expect(mockCreate).toHaveBeenCalledWith('What is machine learning', 's1')
  })

  it('duplicateSession copies session with new id', async () => {
    const opts = defaultOpts()
    const session = { id: 's1', name: 'Chat', messages: [], createdAt: '2024-01-01', updatedAt: '2024-01-01', synced: false, starred: false, pinned: false }
    mockLoadSession.mockResolvedValue(session)
    mockLoadSessions.mockResolvedValue([session])
    const { result } = renderHook(() => useChatSessions(opts))
    await act(async () => { await result.current.duplicateSession('s1') })
    const saved = mockSaveSession.mock.calls[0][0]
    expect(saved.name).toBe('Chat (copy)')
    expect(saved.id).not.toBe('s1')
  })

  it('starSession updates starred state', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    await act(async () => { await result.current.starSession('s1', true) })
    expect(mockUpdateSessionFn).toHaveBeenCalledWith('s1', { starred: true })
    expect(opts.showToast).toHaveBeenCalledWith('Conversation starred')
  })

  it('pinSession updates pinned state', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    await act(async () => { await result.current.pinSession('s1', true) })
    expect(mockUpdateSessionFn).toHaveBeenCalledWith('s1', { pinned: true })
  })

  it('renameSession updates name', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    await act(async () => { await result.current.renameSession('s1', 'New Name') })
    expect(mockUpdateSessionFn).toHaveBeenCalledWith('s1', { name: 'New Name' })
    expect(opts.showToast).toHaveBeenCalledWith('Conversation renamed')
  })

  it('deleteSession removes session and clears messages if current', async () => {
    mockGetKV.mockResolvedValueOnce('s1')
    mockLoadSessions.mockResolvedValue([])
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    await act(async () => { await result.current.deleteSession('s1') })
    expect(mockDeleteSession).toHaveBeenCalledWith('s1')
    expect(mockDelete).toHaveBeenCalledWith('s1')
    expect(opts.setMessages).toHaveBeenCalledWith([])
    expect(opts.setSessionSaved).toHaveBeenCalledWith(false)
  })

  it('sidebarConversations maps sessions correctly', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))
    act(() => {
      result.current.setSessions([{
        id: 's1', name: 'Test Chat', messages: [{ id: 'm1', role: 'user', content: 'hi', timestamp: new Date() }],
        createdAt: '2024-01-01', updatedAt: '2024-01-02', synced: true, starred: true, pinned: false,
      }])
    })
    expect(result.current.sidebarConversations).toHaveLength(1)
    expect(result.current.sidebarConversations[0].name).toBe('Test Chat')
    expect(result.current.sidebarConversations[0].starred).toBe(true)
    expect(result.current.sidebarConversations[0].message_count).toBe(1)
  })
})

describe('useChatSessions.loadSession', () => {
  const localSession = {
    id: 's1',
    name: 'Test chat',
    messages: [
      { id: 'm1', role: 'user' as const, content: 'hello', timestamp: new Date() },
    ],
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
    synced: false,
    starred: false,
    pinned: false,
  }

  const defaultOpts = () => ({
    setMessages: vi.fn(),
    setInput: vi.fn(),
    setSessionSaved: vi.fn(),
    setSessionLoading: vi.fn(),
    sessionIdRef: { current: 's1' } as React.MutableRefObject<string>,
    showToast: vi.fn(),
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mockLoadSession.mockResolvedValue(localSession)
    mockGetDraft.mockResolvedValue('')
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('merges remote messages with local and releases sessionLoading', async () => {
    mockFetchMessages.mockResolvedValue([{ role: 'assistant', content: 'remote reply' }])
    const opts = { ...defaultOpts(), sessionIdRef: { current: 's9' } as React.MutableRefObject<string> }
    const { result } = renderHook(() => useChatSessions(opts))

    await act(async () => { await result.current.loadSession('s1') })

    expect(opts.sessionIdRef.current).toBe('s1')
    expect(opts.setSessionLoading).toHaveBeenNthCalledWith(1, true)
    const contents = opts.setMessages.mock.calls[0][0].map((m: { content: string }) => m.content)
    expect(contents).toContain('remote reply')
    expect(contents).toContain('hello')
    expect(opts.setSessionLoading).toHaveBeenLastCalledWith(false)
    expect(opts.showToast).toHaveBeenCalledWith('Loaded: Test chat')
    expect(mockFetchMessages).toHaveBeenCalledWith('s1', expect.objectContaining({ silent: true, signal: expect.any(AbortSignal) }))
  })

  it('falls back to local messages when the remote merge hangs past the timeout', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))

    vi.useFakeTimers()
    let capturedSignal: AbortSignal | undefined
    mockFetchMessages.mockImplementation((_id: string, reqOpts?: { signal?: AbortSignal }) => {
      capturedSignal = reqOpts?.signal
      return new Promise<never>(() => {})
    })
    let settled = false
    const load = result.current.loadSession('s1').then(() => { settled = true })
    await vi.advanceTimersByTimeAsync(8001)
    await act(async () => { await load })

    expect(settled).toBe(true)
    expect(capturedSignal?.aborted).toBe(true)
    expect(opts.setMessages).toHaveBeenCalledWith(localSession.messages)
    expect(opts.setSessionLoading).toHaveBeenLastCalledWith(false)
    expect(opts.showToast).toHaveBeenCalledWith('Loaded: Test chat')
  })

  it('falls back to local messages when the remote merge rejects', async () => {
    mockFetchMessages.mockRejectedValue(new Error('server down'))
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))

    await act(async () => { await result.current.loadSession('s1') })

    expect(opts.setMessages).toHaveBeenCalledWith(localSession.messages)
    expect(opts.setSessionLoading).toHaveBeenLastCalledWith(false)
  })

  it('drops stale remote results when the session changes while loading', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))

    let resolveRemote: (v: Array<{ role: string; content: string }>) => void = () => {}
    mockFetchMessages.mockImplementation(
      () => new Promise<Array<{ role: string; content: string }>>(res => { resolveRemote = res }),
    )
    const load = result.current.loadSession('s1')
    await vi.waitFor(() => expect(mockFetchMessages).toHaveBeenCalled())

    opts.sessionIdRef.current = 's2'

    await act(async () => {
      resolveRemote([{ role: 'assistant', content: 'stale remote reply' }])
      await load
    })

    expect(opts.setMessages).not.toHaveBeenCalled()
    expect(opts.setSessionLoading).toHaveBeenCalledTimes(1)
    expect(opts.setSessionLoading).toHaveBeenCalledWith(true)
  })

  it('does not release loading or apply messages when a stale load eventually times out', async () => {
    const opts = defaultOpts()
    const { result } = renderHook(() => useChatSessions(opts))

    vi.useFakeTimers()
    mockFetchMessages.mockImplementation(() => new Promise<never>(() => {}))
    const load = result.current.loadSession('s1')
    await act(async () => {})

    opts.sessionIdRef.current = 's2'

    await vi.advanceTimersByTimeAsync(8001)
    await act(async () => { await load })

    expect(opts.setMessages).not.toHaveBeenCalled()
    expect(opts.setSessionLoading).toHaveBeenCalledTimes(1)
    expect(opts.setSessionLoading).toHaveBeenCalledWith(true)
  })
})
