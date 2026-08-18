// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import {
  ChatToolbarSection,
  ChatSettingsSection,
  ChatSearchSection,
  ChatDialogSection,
} from './ChatPageSections'
import type { ChatPageController } from './hooks/useChatPageController'

vi.mock('next/dynamic', () => ({
  default: (importFn: () => Promise<any>, _opts?: any) => {
    const Stub = (props: any) => {
      const path = importFn.toString()
      // Extract last segment from import path like () => import('@/features/chat/components/dialogs/ChatSettings').then(...)
      const match = path.match(/\/([A-Za-z]+)(?:\.)?'/) || path.match(/\/([A-Za-z]+)`/)
      const name = match?.[1] ?? 'Dynamic'
      return <div data-testid={`dyn-${name}`} />
    }
    Stub.displayName = 'DynamicStub'
    return Stub
  },
}))

vi.mock('@/features/chat/components/toolbar/ChatToolbar', () => ({
  ChatToolbar: () => <div data-testid="chat-toolbar" />,
}))

vi.mock('@/features/chat/components/toolbar/ModeBar', () => ({
  ModeBar: () => <div data-testid="mode-bar" />,
}))

vi.mock('@/features/chat/components', () => ({
  ChatArea: () => <div data-testid="chat-area" />,
  ErrorBanner: () => <div data-testid="error-banner" />,
}))

vi.mock('@/features/chat/components/layout/ImageDropZone', () => ({
  ImageDropZone: ({ children }: any) => <div data-testid="image-drop-zone">{children}</div>,
}))

vi.mock('@/features/chat/contexts/ChatToolbarContext', () => ({
  ChatToolbarProvider: ({ children }: any) => <div>{children}</div>,
}))

vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: { update: vi.fn() },
}))

function makeController(overrides?: Partial<ChatPageController>): ChatPageController {
  const base = {
    chat: {
      messages: [],
      loading: false,
      sessionLoading: false,
      sessionIdRef: { current: 's1' },
      sidebarConversations: [],
      archivedCount: 0,
      input: '',
      images: [],
      toolEvents: [],
      ragVerification: null,
      currentError: null,
      loadingRef: { current: null },
      setInput: vi.fn(),
      setMessages: vi.fn(),
      setLoading: vi.fn(),
      setCurrentError: vi.fn(),
      sendMessage: vi.fn(),
      loadSession: vi.fn(),
      newChat: vi.fn(),
      deleteSession: vi.fn(),
      starSession: vi.fn(),
      pinSession: vi.fn(),
      archiveSession: vi.fn(),
      renameSession: vi.fn(),
      duplicateSession: vi.fn(),
      handleCopy: vi.fn(),
      handleRegenerate: vi.fn(),
      handleThumbsUp: vi.fn(),
      handleThumbsDown: vi.fn(),
      handleEditMessage: vi.fn(),
      handleSuggestionClick: vi.fn(),
      handleRetry: vi.fn(),
      handleAddImage: vi.fn(),
      handleRemoveImage: vi.fn(),
    },
    ui: {
      showSettings: false,
      sidebarOpen: false,
      showConversationViewer: false,
      showConversationSearch: false,
      toolPanelOpen: false,
      voiceMode: false,
      searchQuery: '',
      setSidebarOpen: vi.fn(),
      setShowConversationViewer: vi.fn(),
      setShowConversationSearch: vi.fn(),
      setToolPanelOpen: vi.fn(),
      setVoiceMode: vi.fn(),
    },
    model: {
      model: 'gpt2',
      temperature: 0.7,
      maxTokens: 256,
      availableModels: [],
      pendingDownload: null,
      modelInfoMap: {},
      setModel: vi.fn(),
      setTemperature: vi.fn(),
      setMaxTokens: vi.fn(),
      setPendingDownload: vi.fn(),
      startDownloadFlowRef: { current: vi.fn() },
    },
    health: null,
    suggestions: [],
    refreshHealth: vi.fn(),
    showToast: vi.fn(),
    clearChat: vi.fn(),
    chatMode: 'chat',
    setChatMode: vi.fn(),
    writeTone: '',
    setWriteTone: vi.fn(),
    writeType: '',
    setWriteType: vi.fn(),
    decideStructure: '',
    setDecideStructure: vi.fn(),
    explainDifficulty: '',
    setExplainDifficulty: vi.fn(),
    translateLangPair: '',
    setTranslateLangPair: vi.fn(),
    brainstormTopic: '',
    setBrainstormTopic: vi.fn(),
    wellnessType: '',
    setWellnessType: vi.fn(),
    createStyle: '',
    setCreateStyle: vi.fn(),
    readFileData: null,
    setReadFileData: vi.fn(),
    readLoading: false,
    handleReadFile: vi.fn(),
    handleWriteSend: vi.fn(),
    handleExecuteCommand: vi.fn(),
    handleImageDropped: vi.fn(),
    handleTextDropped: vi.fn(),
    handlePDFDropped: vi.fn(),
    isBookmarked: vi.fn().mockReturnValue(false),
    handleToggleBookmark: vi.fn(),
    handleDeleteMessage: vi.fn(),
    handleSaveToKnowledge: vi.fn(),
    collapsibleLength: undefined,
    toolbarValue: {} as any,
    bookmarks: [],
    removeBookmark: vi.fn(),
    clearAll: vi.fn(),
    systemPromptOpen: false,
    setSystemPromptOpen: vi.fn(),
    customSystemPrompt: '',
    handleSaveSystemPrompt: vi.fn(),
    convCollapsed: false,
    toggleConv: vi.fn(),
  }
  return base as unknown as ChatPageController
}

describe('ChatToolbarSection', () => {
  it('renders ChatToolbar', () => {
    render(<ChatToolbarSection controller={makeController()} />)
    expect(document.querySelector('[data-testid="chat-toolbar"]')).toBeTruthy()
  })
})

describe('ChatSettingsSection', () => {
  it('returns null when showSettings is false', () => {
    const { container } = render(<ChatSettingsSection controller={makeController()} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders settings when showSettings is true', () => {
    const ctrl = makeController()
    ;(ctrl.ui as any).showSettings = true
    const { container } = render(<ChatSettingsSection controller={ctrl} />)
    expect(container.querySelector('[data-testid^="dyn-"]')).toBeTruthy()
  })
})

describe('ChatSearchSection', () => {
  it('renders viewer and search stubs', () => {
    const { container } = render(<ChatSearchSection controller={makeController()} />)
    const dyns = container.querySelectorAll('[data-testid^="dyn-"]')
    expect(dyns.length).toBeGreaterThanOrEqual(2)
  })
})

describe('ChatDialogSection', () => {
  it('returns empty when no dialogs are open', () => {
    const { container } = render(<ChatDialogSection controller={makeController()} />)
    expect(container.textContent).toBe('')
  })

  it('renders ChatToolPanel when toolPanelOpen', () => {
    const ctrl = makeController()
    ;(ctrl.ui as any).toolPanelOpen = true
    const { container } = render(<ChatDialogSection controller={ctrl} />)
    expect(container.querySelector('[data-testid^="dyn-"]')).toBeTruthy()
  })

  it('renders DownloadDialog when pendingDownload is set', () => {
    const ctrl = makeController()
    ;(ctrl.model as any).pendingDownload = { modelId: 'gpt2', info: {} }
    const { container } = render(<ChatDialogSection controller={ctrl} />)
    expect(container.querySelector('[data-testid^="dyn-"]')).toBeTruthy()
  })

  it('renders VoiceChatMode when voiceMode is true', () => {
    const ctrl = makeController()
    ;(ctrl.ui as any).voiceMode = true
    const { container } = render(<ChatDialogSection controller={ctrl} />)
    expect(container.querySelector('[data-testid^="dyn-"]')).toBeTruthy()
  })

  it('renders SystemPromptDialog when systemPromptOpen', () => {
    const ctrl = makeController()
    ;(ctrl as any).systemPromptOpen = true
    const { container } = render(<ChatDialogSection controller={ctrl} />)
    expect(container.querySelector('[data-testid^="dyn-"]')).toBeTruthy()
  })
})
