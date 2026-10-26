/**
 * chat-store persist regression (card 7f64b8fa-9fd2-474c-b3f3-cc7b999f1dd7).
 *
 * Bug: send() captured `messagesWithNew` BEFORE the stream started, and the
 * stream-complete persist saved that capture — token flushes REPLACE message
 * objects in live state, so the saved transcript ended with an empty
 * assistant placeholder ("[.., prompt, '']"). The real reply only reached
 * /session/{sid}/context on a LATER save. These tests prove one mocked
 * exchange persists the FULL assistant reply from live state at complete
 * time, with no later save, and without changing request payload construction.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { streamChatResponse } from '@/lib/stream-chat-response'
import { chatController } from '@/lib/chat-controller'
import { chatDB } from '@/lib/db'
import { useChatStore, type ChatConfig } from './chat-store'

vi.mock('@/lib/stream-chat-response', () => ({
  streamChatResponse: vi.fn(),
}))

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    saveSessionContext: vi.fn(async () => {}),
    cancelStream: vi.fn(async () => {}),
    approveTool: vi.fn(async () => {}),
    injectContext: vi.fn(async () => {}),
  },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    context: vi.fn(async () => ({ count: 0, context: '' })),
  },
}))

vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    trainImage: vi.fn(async () => ({})),
    getCapabilities: vi.fn(async () => ({})),
    getTrainingReport: vi.fn(async () => ({ caption_history: [], vocab_size: 0 })),
  },
}))

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn(async () => null),
    setKV: vi.fn(async () => {}),
    getDraft: vi.fn(async () => null),
    saveDraft: vi.fn(async () => {}),
    deleteDraft: vi.fn(async () => {}),
    saveSession: vi.fn(async () => {}),
    loadSessions: vi.fn(async () => []),
    deleteSession: vi.fn(async () => {}),
  },
}))

vi.mock('@/lib/store', () => ({
  useAppStore: {
    getState: vi.fn(() => ({
      settings: { customContext: '', autoApproveTools: true },
      injectedKnowledge: [],
    })),
    setState: vi.fn(),
    subscribe: vi.fn(() => () => {}),
  },
  getKnowledgeContext: vi.fn(async () => ''),
}))

vi.mock('@/lib/memory-events', () => ({
  publishMemoryEvent: vi.fn(),
}))

vi.mock('@/features/chat/components/feedback/ErrorBanner', () => ({
  getErrorInfo: (status: number, text: string) => ({ status, text }),
}))

// lib/*.test.ts runs in the `node` environment — no requestAnimationFrame.
// The store schedules token flushes through rAF; a 16ms timer shim keeps the
// production flush path intact (and lets the test prove that the complete-time
// save does NOT depend on a timer firing: no macrotask runs before assert).
if (typeof globalThis.requestAnimationFrame !== 'function') {
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) =>
    setTimeout(() => cb(0), 16)) as unknown as typeof requestAnimationFrame
  globalThis.cancelAnimationFrame = ((id: number) =>
    clearTimeout(id as unknown as ReturnType<typeof setTimeout>)) as unknown as typeof cancelAnimationFrame
}

const makeConfig = (): ChatConfig => ({
  model: 'test-model',
  temperature: 0.7,
  maxTokens: 64,
  currentSoul: null,
  currentAgent: null,
  useLocalEngine: false,
  engineRef: { current: null } as unknown as ChatConfig['engineRef'],
  engineLoadingRef: { current: false },
  initLocalEngine: async () => false,
  customSystemPrompt: '',
  showToast: vi.fn(),
  recordFeedback: vi.fn(async () => true),
  onVisionUpdate: vi.fn(),
  onKnowledgeUpdate: vi.fn(),
})

const saveSessionContext = vi.mocked(chatController.saveSessionContext)

beforeEach(() => {
  useChatStore.setState({
    _config: makeConfig(),
    sessionId: 'sess-1',
    userId: 'u1',
    messages: [],
    input: '',
    images: [],
    loading: false,
    currentError: null,
    streamingMessageId: null,
    toolEvents: [],
    ragVerification: null,
    contextLayers: [],
    pendingToolApproval: null,
    sessionSaved: false,
  })

  // One mocked exchange: two tokens, then complete — mirroring
  // stream-chat-response, which fires onComplete before the stream promise
  // resolves. Tokens sit in the rAF buffer until complete → cancelFlush().
  vi.mocked(streamChatResponse).mockImplementation(async (params) => {
    params.onToken('Hello')
    params.onToken(' world')
    params.onComplete()
  })
})

describe('chatStore sendMessage — stream-complete persist', () => {
  it('persists the FULL assistant reply at complete time, not the pre-stream placeholder', async () => {
    await useChatStore.getState().sendMessage('hello there')

    // Exactly one context save per exchange — no later save needed.
    expect(saveSessionContext).toHaveBeenCalledTimes(1)
    const [sessionId, persisted] = saveSessionContext.mock.calls[0]
    expect(sessionId).toBe('sess-1')
    expect(persisted).toHaveLength(2)
    expect(persisted[0]).toEqual({ role: 'user', content: 'hello there' })
    // The regression: this used to be '' (the empty assistant placeholder
    // captured before the stream started).
    expect(persisted[1]).toEqual({ role: 'assistant', content: 'Hello world' })

    // Ordering proof — 'complete' fires AFTER the final flush has written the
    // finished message into state: onComplete → cancelFlush() flushes the rAF
    // buffer synchronously, and no timer can run during this microtask-only
    // path. If the final replacement could land after the save, the persisted
    // transcript above would still be the placeholder.
    const live = useChatStore.getState().messages
    expect(live).toHaveLength(2)
    expect(live[1].content).toBe('Hello world')
    expect(persisted).toEqual(live.map((m) => ({ role: m.role, content: m.content })))

    // Scope guard: request payload construction is unchanged — the empty
    // placeholder is still filtered out of the SSE request body.
    const streamParams = vi.mocked(streamChatResponse).mock.calls[0][0]
    expect(streamParams.messages).toEqual([{ role: 'user', content: 'hello there' }])
  })

  it('second exchange persists a full 4-message transcript — replace, no duplication', async () => {
    await useChatStore.getState().sendMessage('hello there')
    await useChatStore.getState().sendMessage('again')

    // One save per exchange, full-array replace semantics each time.
    expect(saveSessionContext).toHaveBeenCalledTimes(2)
    const [, secondPersist] = saveSessionContext.mock.calls[1]
    expect(secondPersist).toEqual([
      { role: 'user', content: 'hello there' },
      { role: 'assistant', content: 'Hello world' },
      { role: 'user', content: 'again' },
      { role: 'assistant', content: 'Hello world' },
    ])

    const live = useChatStore.getState().messages
    expect(secondPersist).toEqual(live.map((m) => ({ role: m.role, content: m.content })))
  })
})

/**
 * Split-brain regression (blank chat pane).
 *
 * Bug: chat-store `init()` scheduled its OWN loadSession via setTimeout, on top
 * of the controller's session load mirrored in by useChatStoreBridge. The
 * controller is the source of truth; the store renders. The rogue store-side load
 * resolved later with a stale/persisted read and clobbered live store.messages —
 * blanking the pane while the toolbar (controller) still showed the real count.
 *
 * Fix: init() no longer loads the session. These tests prove a stale persisted
 * read arriving after init does NOT overwrite live messages the bridge pushed.
 */
describe('init() does not race the controller bridge (split-brain)', () => {
  it('does not schedule an independent session load on init', async () => {
    const loadSessions = vi.mocked(chatDB.loadSessions)
    loadSessions.mockClear()

    // Bridge has just mirrored live controller messages into the store.
    useChatStore.setState({
      messages: [
        { id: 'm1', role: 'user', content: 'live user msg', timestamp: new Date() },
        { id: 'm2', role: 'assistant', content: 'live assistant reply', timestamp: new Date() },
      ],
    })

    // A persisted session exists (stale, empty messages) — the old racer would
    // find it and overwrite the live pane with [].
    vi.mocked(chatDB.getKV).mockResolvedValue('sess-1' as never)
    loadSessions.mockResolvedValue([
      { id: 'sess-1', name: 'stale', messages: [], updatedAt: '2026-01-01' },
    ] as never)

    useChatStore.getState().init(makeConfig())

    // Flush the macrotask the racer used (setTimeout 0) plus microtasks.
    await new Promise((resolve) => setTimeout(resolve, 5))
    await Promise.resolve()

    // The store never issued its own load, and live messages survive intact.
    expect(loadSessions).not.toHaveBeenCalled()
    const msgs = useChatStore.getState().messages
    expect(msgs).toHaveLength(2)
    expect(msgs[0].content).toBe('live user msg')
    expect(msgs[1].content).toBe('live assistant reply')
  })

  it('init() still hydrates sessionId without touching messages', async () => {
    useChatStore.setState({ messages: [], sessionId: '' })
    vi.mocked(chatDB.getKV).mockResolvedValue('sess-9' as never)

    useChatStore.getState().init(makeConfig())
    await Promise.resolve()
    await Promise.resolve()

    expect(useChatStore.getState().sessionId).toBe('sess-9')
    expect(useChatStore.getState().messages).toEqual([])
  })
})
