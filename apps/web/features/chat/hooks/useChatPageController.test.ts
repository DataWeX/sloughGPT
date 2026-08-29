import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'

const { mockUseRouter, mockUseSearchParams, mockHealthLegacy } = vi.hoisted(() => ({
  mockUseRouter: { push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() },
  mockUseSearchParams: { get: vi.fn() },
  mockHealthLegacy: null as any,
}))

vi.mock('next/navigation', () => ({
  useRouter: () => mockUseRouter,
  useSearchParams: () => mockUseSearchParams,
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({ healthLegacy: mockHealthLegacy }),
}))

vi.mock('@/lib/controllers', () => ({
  soulsController: { switch: vi.fn() },
  multimodalController: { uploadPDF: vi.fn() },
  modelController: { list: vi.fn().mockResolvedValue([]) },
}))

vi.mock('@/lib/chat-commands', () => ({
  ChatCommand: {},
}))

vi.mock('@/features/chat/hooks/useChatUI', () => ({
  useChatUI: () => ({
    showSettings: false, toggleSettings: vi.fn(), setShowSettings: vi.fn(),
    searchQuery: '', setSearchQuery: vi.fn(), handleSearchChange: vi.fn(),
    handleSearchClear: vi.fn(), matchIndex: 0, setMatchIndex: vi.fn(),
    showConversationViewer: false, setShowConversationViewer: vi.fn(),
    showConversationSearch: false, setShowConversationSearch: vi.fn(),
    toolPanelOpen: true, setToolPanelOpen: vi.fn(),
    voiceMode: false, setVoiceMode: vi.fn(),
    sidebarOpen: false, setSidebarOpen: vi.fn(),
    searchInputRef: { current: null },
    chatScreenRef: { current: null },
  }),
}))

vi.mock('@/features/chat/hooks/useChatVision', () => ({
  useChatVision: () => ({
    visionCaps: null, setVisionCaps: vi.fn(),
    visionCaptionHistory: [], setVisionCaptionHistory: vi.fn(),
    visionVocabSize: 0, setVisionVocabSize: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatAgents', () => ({
  useChatAgents: () => ({
    currentAgent: null, setCurrentAgent: vi.fn(),
    agents: [], fetchInitialData: vi.fn(),
    knowledgeCtx: { count: 0, context: '' }, setKnowledgeCtx: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatLocalEngine', () => ({
  useChatLocalEngine: () => ({
    useLocalEngine: false, engineRef: { current: null },
    engineLoadingRef: { current: false }, initLocalEngine: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatModelSettings', () => ({
  useChatModelSettings: () => ({
    model: 'gpt2', setModel: vi.fn().mockResolvedValue(undefined),
    temperature: 0.8, setTemperature: vi.fn(),
    maxTokens: 200, currentSoul: null,
    fetchInitialData: vi.fn(),
  }),
}))

vi.mock('@/lib/store', () => ({
  useSettings: () => ({ collapsibleMessageLength: 2000 }),
}))

vi.mock('@/lib/feedback-store', () => ({
  useFeedbackStore: () => ({
    recordFeedback: vi.fn(),
    fetchStats: vi.fn(),
    fetchAdapterStats: vi.fn(),
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: { getState: vi.fn(() => ({ addToast: vi.fn() })) },
}))

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn().mockResolvedValue(undefined),
    setKV: vi.fn().mockResolvedValue(undefined),
  },
}))

vi.mock('@/features/chat/hooks/useChatBookmarks', () => ({
  useChatBookmarks: () => ({
    bookmarks: [], addBookmark: vi.fn(), removeBookmark: vi.fn(),
    isBookmarked: vi.fn(() => false), clearAll: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatMessages', () => ({
  useChatMessages: (config: any) => ({
    messages: [], setMessages: vi.fn(),
    input: '', setInput: vi.fn(),
    loading: false, setLoading: vi.fn(),
    currentError: null, setCurrentError: vi.fn(),
    loadingRef: { current: false }, newChatRef: { current: vi.fn() },
    handleRegenerateRef: { current: vi.fn() },
    sendMessage: vi.fn(), sessionIdRef: { current: null },
    loadSession: vi.fn(), renameSession: vi.fn(), duplicateSession: vi.fn(),
    handleExportMarkdown: vi.fn(), handleAddImage: vi.fn(),
    newChat: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatMode', () => ({
  useChatMode: () => ({
    chatMode: 'chat', setChatMode: vi.fn(),
    writeTone: 'Friendly', setWriteTone: vi.fn(),
    writeType: 'Email', setWriteType: vi.fn(),
    rewriteStyle: 'Fix Grammar', setRewriteStyle: vi.fn(),
    decideStructure: 'Pros & Cons', setDecideStructure: vi.fn(),
    explainDifficulty: 'Beginner', setExplainDifficulty: vi.fn(),
    translateLangPair: 'English→Spanish', setTranslateLangPair: vi.fn(),
    brainstormTopic: 'Ideas', setBrainstormTopic: vi.fn(),
    wellnessType: 'Meditation', setWellnessType: vi.fn(),
    createStyle: 'Photorealistic', setCreateStyle: vi.fn(),
  }),
}))

vi.mock('@/features/chat/hooks/useChatKeyboard', () => ({
  useChatKeyboard: vi.fn(),
}))

vi.mock('@/features/chat/hooks/useChatToolbarValue', () => ({
  useChatToolbarValue: () => ({}),
}))

vi.mock('@/features/chat/hooks/useChatContextValue', () => ({
  useChatHealthValue: () => ({}),
  useChatModelValue: () => ({}),
  useChatUIValue: () => ({}),
}))

vi.mock('@/features/chat/contexts/ConvSidebarContext', () => ({
  useConvSidebar: () => ({
    setOpen: vi.fn(), convCollapsed: false, toggleConv: vi.fn(),
  }),
}))

vi.mock('@/lib/chat-utils', () => ({
  computeSearchMatches: vi.fn(() => ({ matchIds: [], matchCount: 0 })),
}))

vi.mock('@/lib/chat-controller', () => ({
  chatController: { getSuggestions: vi.fn().mockResolvedValue([]) },
}))

vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: {},
}))

vi.mock('@/lib/images-controller', () => ({
  imagesController: { generate: vi.fn() },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: { add: vi.fn() },
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: vi.fn(() => 'error'),
  formatToastError: vi.fn(() => 'formatted error'),
}))

vi.mock('@/lib/format-bytes', () => ({
  PDF_ANALYSIS_MAX_TOKENS: 4096,
}))

vi.mock('@/features/chat/components/input/ImageUpload', () => ({
  resizeImage: vi.fn(),
}))

vi.mock('@/lib/dev-log', () => ({
  devDebug: vi.fn(),
  logger: { child: vi.fn(() => ({ info: vi.fn(), warning: vi.fn(), error: vi.fn() })) },
}))

import { useChatPageController } from './useChatPageController'

function renderController() {
  return renderHook(() =>
    useChatPageController(vi.fn(), vi.fn().mockResolvedValue(undefined))
  )
}

describe('useChatPageController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('returns expected interface keys', () => {
    const { result } = renderController()
    expect(result.current).toHaveProperty('health')
    expect(result.current).toHaveProperty('ui')
    expect(result.current).toHaveProperty('chat')
    expect(result.current).toHaveProperty('model')
    expect(result.current).toHaveProperty('agents')
    expect(result.current).toHaveProperty('vision')
    expect(result.current).toHaveProperty('engine')
    expect(result.current).toHaveProperty('chatMode')
    expect(result.current).toHaveProperty('clearChat')
    expect(result.current).toHaveProperty('toolbarValue')
    expect(result.current).toHaveProperty('healthValue')
    expect(result.current).toHaveProperty('modelValue')
    expect(result.current).toHaveProperty('uiValue')
  })

  it('returns initial chatMode as chat', () => {
    const { result } = renderController()
    expect(result.current.chatMode).toBe('chat')
  })

  it('returns initial suggestions as empty', () => {
    const { result } = renderController()
    expect(result.current.suggestions).toEqual([])
  })

  it('returns initial bookmarks as empty', () => {
    const { result } = renderController()
    expect(result.current.bookmarks).toEqual([])
  })

  it('returns clearChat as a function', () => {
    const { result } = renderController()
    expect(typeof result.current.clearChat).toBe('function')
  })

  it('returns handleWriteSend as a function', () => {
    const { result } = renderController()
    expect(typeof result.current.handleWriteSend).toBe('function')
  })

  it('returns handleToggleBookmark as a function', () => {
    const { result } = renderController()
    expect(typeof result.current.handleToggleBookmark).toBe('function')
  })

  it('returns handleDeleteMessage as a function', () => {
    const { result } = renderController()
    expect(typeof result.current.handleDeleteMessage).toBe('function')
  })

  it('returns readFileData as null initially', () => {
    const { result } = renderController()
    expect(result.current.readFileData).toBeNull()
  })

  it('returns readLoading as false initially', () => {
    const { result } = renderController()
    expect(result.current.readLoading).toBe(false)
  })

  it('returns customSystemPrompt as empty string initially', () => {
    const { result } = renderController()
    expect(result.current.customSystemPrompt).toBe('')
  })

  it('returns systemPromptOpen as false initially', () => {
    const { result } = renderController()
    expect(result.current.systemPromptOpen).toBe(false)
  })

  it('returns modelDescriptions as empty object initially', () => {
    const { result } = renderController()
    expect(result.current.modelDescriptions).toEqual({})
  })

  it('returns function handlers for execute command', () => {
    const { result } = renderController()
    expect(typeof result.current.handleExecuteCommand).toBe('function')
  })

  it('returns function handlers for save to knowledge', () => {
    const { result } = renderController()
    expect(typeof result.current.handleSaveToKnowledge).toBe('function')
  })

  it('returns function handlers for read file', () => {
    const { result } = renderController()
    expect(typeof result.current.handleReadFile).toBe('function')
  })

  it('returns function handlers for image dropped', () => {
    const { result } = renderController()
    expect(typeof result.current.handleImageDropped).toBe('function')
  })

  it('returns function handlers for text dropped', () => {
    const { result } = renderController()
    expect(typeof result.current.handleTextDropped).toBe('function')
  })

  it('returns function handlers for PDF dropped', () => {
    const { result } = renderController()
    expect(typeof result.current.handlePDFDropped).toBe('function')
  })

  it('returns function handlers for save system prompt', () => {
    const { result } = renderController()
    expect(typeof result.current.handleSaveSystemPrompt).toBe('function')
  })

  it('returns function handler for select agent with toast', () => {
    const { result } = renderController()
    expect(typeof result.current.handleSelectAgentWithToast).toBe('function')
  })
})
