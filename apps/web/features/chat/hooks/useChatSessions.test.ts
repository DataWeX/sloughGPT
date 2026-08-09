import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

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
