/**
 * useChatStoreBridge — the store's CONFIG seam (stage 3, card 20261116_130540).
 *
 * The bridge's whole job: `_config` follows every live controller input, and
 * the hook-owned session-list callbacks (saveSessionToStorage / renameSession)
 * route through to the facade. State never flows through here — an init-once
 * config would freeze the send pipeline's inputs at first render.
 */

import { renderHook } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

const { mockRecordFeedback } = vi.hoisted(() => ({ mockRecordFeedback: vi.fn(async () => true) }))

vi.mock('@/lib/feedback-store', () => ({
  useFeedbackStore: () => ({ recordFeedback: mockRecordFeedback }),
}))

import { useChatStoreBridge } from './useChatStoreBridge'
import { useChatStore } from '@/lib/chat-store'
import { chatDB } from '@/lib/db'
import type { ChatPageController } from './useChatPageController'

function makeController(overrides: Record<string, unknown> = {}) {
  return {
    model: {
      model: 'gpt2',
      temperature: 0.7,
      maxTokens: 256,
      currentSoul: null,
    },
    showToast: vi.fn(),
    vision: {
      setVisionCaps: vi.fn(),
      setVisionCaptionHistory: vi.fn(),
      setVisionVocabSize: vi.fn(),
    },
    agents: {
      currentAgent: null,
      setKnowledgeCtx: vi.fn(),
    },
    engine: {
      useLocalEngine: false,
      engineRef: { current: null },
      engineLoadingRef: { current: false },
      initLocalEngine: vi.fn(),
    },
    customSystemPrompt: '',
    chat: {
      saveSessionToStorage: vi.fn(async () => {}),
      renameSession: vi.fn(async () => {}),
    },
    ...overrides,
  } as unknown as ChatPageController
}

describe('useChatStoreBridge — config seam', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.spyOn(chatDB, 'getKV').mockResolvedValue(undefined as never)
    vi.spyOn(chatDB, 'setKV').mockResolvedValue(undefined as never)
    vi.spyOn(chatDB, 'getDraft').mockResolvedValue(undefined as never)
    useChatStore.setState({ _config: null, sessionId: '', input: '', messages: [] })
  })

  it('initializes _config from the controller on first render', () => {
    renderHook(() => useChatStoreBridge(makeController()))
    const config = useChatStore.getState()._config

    expect(config).not.toBeNull()
    expect(config!.model).toBe('gpt2')
    expect(config!.temperature).toBe(0.7)
    expect(config!.maxTokens).toBe(256)
    expect(config!.showToast).toBeDefined()
    expect(config!.recordFeedback).toBe(mockRecordFeedback)
  })

  it('follows controller input changes on rerender (config is never frozen)', () => {
    const { rerender } = renderHook(({ controller }) => useChatStoreBridge(controller), {
      initialProps: { controller: makeController() },
    })
    expect(useChatStore.getState()._config!.model).toBe('gpt2')

    // The user switches models mid-session — the send pipeline must see it.
    // The controller hands the bridge a fresh object (as it does every render
    // where an input changed); the sync effect re-runs and _config follows.
    rerender({
      controller: makeController({
        model: { model: 'llama-7b', temperature: 0.2, maxTokens: 256, currentSoul: null },
      }),
    })

    const config = useChatStore.getState()._config!
    expect(config.model).toBe('llama-7b')
    expect(config.temperature).toBe(0.2)
  })

  it('routes the save seam to the hook-owned chat.saveSessionToStorage', async () => {
    const controller = makeController()
    renderHook(() => useChatStoreBridge(controller))
    const config = useChatStore.getState()._config!

    const msgs = [{ id: 'm1', role: 'user' as const, content: 'hi', timestamp: new Date() }]
    await config.saveSessionToStorage(msgs, 'sess-42')

    expect((controller as any).chat.saveSessionToStorage).toHaveBeenCalledWith(msgs, 'sess-42')
  })

  it('routes auto-naming to the hook-owned chat.renameSession', async () => {
    const controller = makeController()
    renderHook(() => useChatStoreBridge(controller))
    const config = useChatStore.getState()._config!

    await config.renameSession('sess-42', 'A long conversation title')

    expect((controller as any).chat.renameSession).toHaveBeenCalledWith(
      'sess-42',
      'A long conversation title',
    )
  })

  it('wires vision/knowledge callbacks to the controller stores', () => {
    const controller = makeController()
    renderHook(() => useChatStoreBridge(controller))
    const config = useChatStore.getState()._config!

    config.onVisionUpdate({} as never, ['cap-a'], 512)
    expect((controller as any).vision.setVisionCaps).toHaveBeenCalledWith({})
    expect((controller as any).vision.setVisionCaptionHistory).toHaveBeenCalledWith(['cap-a'])
    expect((controller as any).vision.setVisionVocabSize).toHaveBeenCalledWith(512)

    config.onKnowledgeUpdate({ count: 3, context: 'fact fact fact' })
    expect((controller as any).agents.setKnowledgeCtx).toHaveBeenCalled()
  })
})
