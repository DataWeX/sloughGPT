import { renderHook } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: { createFromChat: vi.fn() },
}))

import { useChatToolbarValue } from './useChatToolbarValue'

function makeConfig(overrides = {}) {
  const chat = {
    sidebarConversations: [],
    sessionIdRef: { current: 's1' },
    loadSession: vi.fn(),
    starSession: vi.fn(),
    pinSession: vi.fn(),
    newChat: vi.fn(),
    handleExportMarkdown: vi.fn(),
    handleCopyMarkdown: vi.fn(),
    messages: [],
    loading: false,
  }
  return {
    ui: {
      searchQuery: '', handleSearchChange: vi.fn(), handleSearchClear: vi.fn(),
      matchIndex: 0, showMobileSearch: false, setShowMobileSearch: vi.fn(),
      sidebarOpen: true, setSidebarOpen: vi.fn(), voiceMode: false, setVoiceMode: vi.fn(),
      toolPanelOpen: false, setToolPanelOpen: vi.fn(),
      showSettings: false, setShowSettings: vi.fn(),
      showConversationViewer: false, setShowConversationViewer: vi.fn(),
      showKeyboardShortcuts: false, setShowKeyboardShortcuts: vi.fn(),
      showTrainingPanel: false, setShowTrainingPanel: vi.fn(),
      showMobileSidebar: false, setShowMobileSidebar: vi.fn(),
      showModelDetails: false, setShowModelDetails: vi.fn(),
    },
    vision: {},
    agents: {
      agents: [], currentAgent: null, handleToggleKnowledge: vi.fn(),
      knowledgeCtx: { showing: false, count: 0, context: '' },
      setAgents: vi.fn(), setCurrentAgent: vi.fn(), setKnowledgeCtx: vi.fn(),
      handleSelectAgent: vi.fn(), fetchInitialData: vi.fn(),
    },
    engine: {
      localModelUrl: '', useLocalEngine: false, localEngineLoading: false, localArchInfo: null,
      handleToggleLocalEngine: vi.fn(),
    },
    model: {
      availableModels: [], model: '', loadingModel: null, modelInfoMap: {},
      downloadProgress: {}, handleSelectModel: vi.fn(), handleUnloadModel: vi.fn(),
      souls: [], currentSoul: null, handleSelectSoul: vi.fn(),
      fineTuned: [], fineTunedLoading: false, handleLoadFineTuned: vi.fn(),
    },
    chat,
    health: { model_loaded: true, model_type: 'gpt2', summary: 'Ready' },
    matchCount: 0, matchIds: [],
    handlePrevMatch: vi.fn(), handleNextMatch: vi.fn(),
    handleSelectAgentWithToast: vi.fn(),
    modelDescriptions: {},
    showToast: vi.fn(),
    ...overrides,
  } as any
}

describe('useChatToolbarValue', () => {
  it('returns conversations group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.conversations.conversations).toBe(config.chat.sidebarConversations)
    expect(result.current.conversations.sessionIdRef).toBe(config.chat.sessionIdRef)
    expect(result.current.conversations.onNewChat).toBe(config.chat.newChat)
  })

  it('returns search group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.search.query).toBe(config.ui.searchQuery)
    expect(result.current.search.onChange).toBe(config.ui.handleSearchChange)
    expect(result.current.search.matchIndex).toBe(config.ui.matchIndex)
  })

  it('returns model group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.model.current).toBe(config.model.model)
    expect(result.current.model.loading).toBe(config.model.loadingModel)
    expect(result.current.model.onSelect).toBe(config.model.handleSelectModel)
  })

  it('wires fine-tuned group into model group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.model.fineTuned).toBeDefined()
    expect(result.current.model.fineTuned?.models).toBe(config.model.fineTuned)
    expect(result.current.model.fineTuned?.loading).toBe(config.model.fineTunedLoading)
    expect(result.current.model.fineTuned?.onLoad).toBe(config.model.handleLoadFineTuned)
  })

  it('returns soul group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.soul.souls).toBe(config.model.souls)
    expect(result.current.soul.current).toBe(config.model.currentSoul)
    expect(result.current.soul.onSelect).toBe(config.model.handleSelectSoul)
  })

  it('returns knowledge group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.knowledge.showing).toBe(config.agents.knowledgeCtx.showing)
    expect(result.current.knowledge.count).toBe(config.agents.knowledgeCtx.count)
    expect(result.current.knowledge.onToggle).toBe(config.agents.handleToggleKnowledge)
  })

  it('returns health group', () => {
    const config = makeConfig()
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.health.status).toBe('ok')
    expect(result.current.health.modelLoaded).toBe(true)
    expect(result.current.health.modelType).toBe('gpt2')
  })

  it('returns offline health', () => {
    const config = makeConfig({ health: 'offline' as any, ui: {} as any })
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.health.status).toBe('offline')
    expect(result.current.health.modelLoaded).toBe(false)
  })

  it('returns bookmarkCount in actions group', () => {
    const config = makeConfig({ bookmarkCount: 5 })
    const { result } = renderHook(() => useChatToolbarValue(config))
    expect(result.current.actions.bookmarkCount).toBe(5)
  })
})
