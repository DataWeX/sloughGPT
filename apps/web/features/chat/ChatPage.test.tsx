import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => ({ cn: vi.fn((...args: any[]) => args.join(' ')), Button: ({ children, ...props }: any) => React.createElement('button', props, children) }))

vi.mock('next/dynamic', () => ({
  __esModule: true,
  default: () => (props: Record<string, unknown>) => React.createElement('div', { 'data-testid': 'dynamic' }),
}))

const { mockPush, mockSearchParamsGet, mockSendMessage, mockNewChat, mockLoadSession, mockGetSuggestions, mockModelList, mockGetHealth, mockFetchStats, mockFetchAdapterStats, mockModelFetchInitialData, mockAgentFetchInitialData, mockSetConvSidebarOpen, mockAddToast, mockKnowledgeAdd, mockImageGenerate, mockAddBookmark, mockRemoveBookmark, mockIsBookmarked, mockClipboardWrite } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockSearchParamsGet: vi.fn(),
  mockSendMessage: vi.fn(),
  mockNewChat: vi.fn(),
  mockLoadSession: vi.fn(),
  mockGetSuggestions: vi.fn(),
  mockModelList: vi.fn(),
  mockGetHealth: vi.fn(),
  mockFetchStats: vi.fn(),
  mockFetchAdapterStats: vi.fn(),
  mockModelFetchInitialData: vi.fn(),
  mockAgentFetchInitialData: vi.fn(),
  mockSetConvSidebarOpen: vi.fn(),
  mockAddToast: vi.fn(),
  mockKnowledgeAdd: vi.fn(),
  mockImageGenerate: vi.fn(),
  mockAddBookmark: vi.fn(),
  mockRemoveBookmark: vi.fn(),
  mockIsBookmarked: vi.fn(),
  mockClipboardWrite: vi.fn(),
}))

const state = vi.hoisted(() => ({
  health: null as any,
  chat: { messages: [] as Array<{ id: string; role: string; content: string; timestamp: Date }>, newChat: null as null | (() => void), sidebarConversations: [] as unknown[] },
  mode: { chatMode: 'chat' },
  ui: { showSettings: false, toolPanelOpen: false },
}))

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }), useSearchParams: () => ({ get: mockSearchParamsGet }) }))
vi.mock('@/hooks/useLiveStatus', () => ({ liveStatusStore: { setState: vi.fn(), getState: () => ({ health: state.health, healthLegacy: state.health }) }, useLiveStatus: () => ({ healthLegacy: state.health }) }))
vi.mock('@/lib/controllers', () => ({ soulsController: { switch: vi.fn() }, multimodalController: { uploadPDF: vi.fn() }, modelController: { list: mockModelList, getHealth: mockGetHealth } }))
vi.mock('@/lib/chat-controller', () => ({ chatController: { getSuggestions: mockGetSuggestions } }))
vi.mock('@/lib/generation-config-controller', () => ({ generationConfigController: { update: vi.fn() } }))
vi.mock('@/lib/feedback-store', () => ({ useFeedbackStore: () => ({ recordFeedback: vi.fn(), fetchStats: mockFetchStats, fetchAdapterStats: mockFetchAdapterStats }) }))
vi.mock('@/lib/images-controller', () => ({ imagesController: { generate: mockImageGenerate } }))
vi.mock('@/lib/files-controller', () => ({ filesController: { extract: vi.fn() } }))
vi.mock('@/lib/knowledge-controller', () => ({ knowledgeController: { add: mockKnowledgeAdd } }))
vi.mock('@/lib/store', () => ({ useSettings: () => ({ collapsibleMessageLength: 500 }), useAppStore: Object.assign(vi.fn((selector: any) => selector({ settings: { autoApproveTools: false }, updateSettings: vi.fn() })), { getState: () => ({ settings: { autoApproveTools: false }, updateSettings: vi.fn(), injectedKnowledge: [] }) }) }))
vi.mock('@/lib/db', () => ({ chatDB: { getKV: vi.fn().mockResolvedValue(undefined), setKV: vi.fn().mockResolvedValue(undefined), deleteKV: vi.fn().mockResolvedValue(undefined), addError: vi.fn().mockResolvedValue(undefined), saveMessageNote: vi.fn().mockResolvedValue(undefined), getMessageNotes: vi.fn().mockResolvedValue([]), removeMessageNote: vi.fn().mockResolvedValue(undefined), searchMessageNotes: vi.fn().mockResolvedValue([]) } }))
vi.mock('@/features/chat/contexts/ConvSidebarContext', () => ({ useConvSidebar: () => ({ setOpen: mockSetConvSidebarOpen, convCollapsed: false, toggleConv: vi.fn() }) }))
vi.mock('@/features/chat/contexts/ChatContext', () => ({ ChatProvider: ({ children }: any) => <div data-testid="chat-provider">{children}</div> }))
vi.mock('@/features/chat/contexts/ChatToolbarContext', () => ({ ChatToolbarProvider: ({ children }: any) => <div>{children}</div> }))
vi.mock('@/features/chat/hooks/useChatToolbarValue', () => ({ useChatToolbarValue: () => ({}) }))
vi.mock('@/features/chat/hooks/useChatContextValue', () => ({ useChatHealthValue: () => ({}), useChatModelValue: () => ({}), useChatUIValue: () => ({}) }))
vi.mock('@/features/chat/components/toolbar/ChatToolbar', () => ({ ChatToolbar: () => <div data-testid="chat-toolbar" /> }))
vi.mock('@/features/chat/components/toolbar/ModeBar', () => ({ ModeBar: () => null }))
vi.mock('@/features/chat/components/layout/ImageDropZone', () => ({ ImageDropZone: ({ children }: any) => <div>{children}</div> }))
vi.mock('@/features/chat/components/input/ImageUpload', () => ({ resizeImage: vi.fn() }))
vi.mock('@/features/chat/components', () => ({
  ChatArea: (props: any) => (
    <div data-testid="chat-area">
      <span data-testid="msg-count">{props.messages?.length ?? 0}</span>
      {props.suggestions?.map((s: { text: string }) => <span key={s.text}>{s.text}</span>)}
      <textarea aria-label="Chat input" value={props.value} onChange={(e) => props.onChange?.(e.target.value)} />
      <button data-testid="send-btn" onClick={() => props.onSend?.()}>Send</button>
      <button data-testid="delete-btn" onClick={() => props.onDelete?.('m1')}>Delete</button>
      <button data-testid="bookmark-btn" onClick={() => props.onBookmark?.('m1')}>Bookmark</button>
      <button data-testid="knowledge-btn" onClick={() => props.onSaveToKnowledge?.('m1', 'some content')}>Knowledge</button>
    </div>
  ),
  ErrorBanner: () => null,
}))

vi.mock('@/features/chat/hooks/useChatUI', async () => {
  const React = await import('react')
  return {
    useChatUI: () => {
      const [showSettings, setShowSettings] = React.useState(state.ui.showSettings)
      const [showConversationViewer, setShowConversationViewer] = React.useState(false)
      const [searchQuery, setSearchQuery] = React.useState('')
      const [showConversationSearch, setShowConversationSearch] = React.useState(false)
      const [showMobileSearch, setShowMobileSearch] = React.useState(false)
      const [matchIndex, setMatchIndex] = React.useState(0)
      const [toolPanelOpen, setToolPanelOpen] = React.useState(state.ui.toolPanelOpen)
      const [voiceMode, setVoiceMode] = React.useState(false)
      const [sidebarOpen, setSidebarOpen] = React.useState(true)
      return {
        showSettings, setShowSettings,
        showConversationViewer, setShowConversationViewer,
        searchQuery, setSearchQuery,
        showConversationSearch, setShowConversationSearch,
        showMobileSearch, setShowMobileSearch,
        matchIndex, setMatchIndex,
        toolPanelOpen, setToolPanelOpen,
        voiceMode, setVoiceMode,
        sidebarOpen, setSidebarOpen,
        chatScreenRef: React.useRef(null),
        searchInputRef: React.useRef(null),
        toggleSettings: vi.fn(), handleSearchChange: vi.fn(), handleSearchClear: vi.fn(),
      }
    },
  }
})

vi.mock('@/features/chat/hooks/useChatMode', async () => {
  const React = await import('react')
  return {
    useChatMode: () => {
      const [chatMode, setChatMode] = React.useState(state.mode.chatMode)
      return {
        chatMode, setChatMode,
        writeTone: 'Friendly', setWriteTone: vi.fn(),
        writeType: 'Email', setWriteType: vi.fn(),
        rewriteStyle: 'Fix Grammar', setRewriteStyle: vi.fn(),
        decideStructure: 'Pro/Con', setDecideStructure: vi.fn(),
        explainDifficulty: 'Beginner', setExplainDifficulty: vi.fn(),
        translateLangPair: 'English→Spanish', setTranslateLangPair: vi.fn(),
        brainstormTopic: 'Ideas', setBrainstormTopic: vi.fn(),
        wellnessType: 'Meditation', setWellnessType: vi.fn(),
        createStyle: 'Watercolor', setCreateStyle: vi.fn(),
        placeholder: 'Type...', handleSend: async (readFileData?: { text: string; filename: string } | null) => {
          if (chatMode === 'read' && !readFileData) {
            mockAddToast('Upload a file first, then ask your question', 'info')
            return
          }
          if (chatMode === 'create') {
            const input = (document.querySelector('[aria-label="Chat input"]') as HTMLTextAreaElement)?.value || ''
            if (input) await mockImageGenerate(input, 'watercolor')
            return
          }
          mockSendMessage()
        },
      }
    },
  }
})

vi.mock('@/features/chat/hooks/useChatMessages', async () => {
  const React = await import('react')
  return {
    useChatMessages: () => {
      const [messages, setMessages] = React.useState(state.chat.messages)
      const [input, setInput] = React.useState('')
      const [loading, setLoading] = React.useState(false)
      const [images, setImages] = React.useState([])
      const [currentError, setCurrentError] = React.useState(null)
      return {
        messages, setMessages,
        input, setInput,
        loading, setLoading,
        images, setImages,
        sessionLoading: false,
        sessionSaved: false,
        currentError, setCurrentError,
        toolEvents: [],
        messagesRef: { current: '' },
        loadingRef: { current: null },
        sessionIdRef: { current: 'sess-1' },
        userIdRef: { current: 'u1' },
        handleRegenerateRef: { current: vi.fn() },
        newChatRef: { current: state.chat.newChat },
        sendMessageRef: { current: vi.fn() },
        sendMessage: mockSendMessage,
        newChat: mockNewChat,
        loadSession: mockLoadSession,
        deleteSession: vi.fn(), starSession: vi.fn(), pinSession: vi.fn(),
        archiveSession: vi.fn(), archivedCount: 0,
        renameSession: vi.fn(), duplicateSession: vi.fn(),
        handleRegenerate: vi.fn(), handleThumbsUp: vi.fn(), handleThumbsDown: vi.fn(),
        handleEditMessage: vi.fn(), handleAddImage: vi.fn(), handleRemoveImage: vi.fn(),
        handleCopy: vi.fn(), handleRetry: vi.fn(), handleSuggestionClick: vi.fn(),
        handleExportMarkdown: vi.fn(), handleCopyMarkdown: vi.fn(),
        sidebarConversations: state.chat.sidebarConversations,
      }
    },
  }
})

vi.mock('@/features/chat/hooks/useChatVision', () => ({ useChatVision: () => ({ visionCaps: null, setVisionCaps: vi.fn(), visionCaptionHistory: [], setVisionCaptionHistory: vi.fn(), visionVocabSize: null, setVisionVocabSize: vi.fn(), refreshVision: vi.fn() }) }))
vi.mock('@/features/chat/hooks/useChatAgents', () => ({ useChatAgents: () => ({ agents: [], setAgents: vi.fn(), currentAgent: null, setCurrentAgent: vi.fn(), knowledgeCtx: { showing: false, count: 0, context: '' }, setKnowledgeCtx: vi.fn(), handleSelectAgent: vi.fn(), handleToggleKnowledge: vi.fn(), fetchInitialData: mockAgentFetchInitialData }) }))
vi.mock('@/features/chat/hooks/useChatLocalEngine', () => ({ useChatLocalEngine: () => ({ useLocalEngine: false, setUseLocalEngine: vi.fn(), localEngineLoading: false, setLocalEngineLoading: vi.fn(), localArchInfo: null, setLocalArchInfo: vi.fn(), localModelUrl: '', setLocalModelUrl: vi.fn(), engineRef: { current: null }, engineLoadingRef: { current: false }, initLocalEngine: vi.fn(), handleToggleLocalEngine: vi.fn() }) }))
vi.mock('@/features/chat/hooks/useChatModelSettings', () => ({ useChatModelSettings: () => ({ model: 'gpt2', setModel: vi.fn(), souls: [], setSouls: vi.fn(), temperature: 0.7, setTemperature: vi.fn(), maxTokens: 100, setMaxTokens: vi.fn(), availableModels: [], setAvailableModels: vi.fn(), modelInfoMap: {}, setModelInfoMap: vi.fn(), downloadProgress: {}, setDownloadProgress: vi.fn(), currentSoul: null, setCurrentSoul: vi.fn(), currentCheckpoint: null, setCurrentCheckpoint: vi.fn(), checkpoints: [], setCheckpoints: vi.fn(), loadingModel: false, setLoadingModel: vi.fn(), pendingDownload: null, setPendingDownload: vi.fn(), learnerInfo: null, setLearnerInfo: vi.fn(), learnerTraining: false, setLearnerTraining: vi.fn(), fineTuned: [], fineTunedLoading: false, fetchFineTuned: vi.fn(), handleLoadFineTuned: vi.fn(), pollIntervalRef: { current: null }, startDownloadFlowRef: { current: vi.fn() }, startDownloadFlow: vi.fn(), handleSelectModel: vi.fn(), handleSelectSoul: vi.fn(), handleUnloadModel: vi.fn(), fetchInitialData: mockModelFetchInitialData }) }))
vi.mock('@/features/chat/hooks/useChatKeyboard', () => ({ useChatKeyboard: () => {} }))
vi.mock('@/features/chat/hooks/useChatBookmarks', () => ({ useChatBookmarks: () => ({ bookmarks: [], addBookmark: mockAddBookmark, removeBookmark: mockRemoveBookmark, isBookmarked: mockIsBookmarked, clearAll: vi.fn() }) }))

const { useToastStore, toastState } = vi.hoisted(() => {
  const toastState = { toasts: [], addToast: () => {} }
  return { useToastStore: { getState: () => toastState }, toastState }
})

vi.mock('@/lib/toast-store', () => ({ useToastStore }))

import ChatPage from './ChatPage'

afterEach(() => { cleanup(); delete (navigator as any).clipboard })
beforeEach(() => {
  vi.clearAllMocks()
  state.health = { model_loaded: true, model_type: 'gpt2' }
  state.chat.messages = []
  state.chat.newChat = mockNewChat
  state.chat.sidebarConversations = []
  state.mode.chatMode = 'chat'
  state.ui.showSettings = false
  state.ui.toolPanelOpen = false
  mockSearchParamsGet.mockReturnValue(null)
  mockModelList.mockResolvedValue([{ id: 'gpt2', description: 'A small model' }])
  mockGetSuggestions.mockResolvedValue([{ text: 'Tell me a story', icon: '🧠' }])
  mockIsBookmarked.mockReturnValue(false)
  mockImageGenerate.mockResolvedValue({ image: 'data:image/png;base64,AAAA' })
  Object.defineProperty(navigator, 'clipboard', { value: { writeText: mockClipboardWrite }, configurable: true })
  mockClipboardWrite.mockResolvedValue(undefined)
  useToastStore.getState().addToast = mockAddToast
})

async function renderChat() {
  render(<ChatPage />)
  await act(async () => {})
}

describe('ChatPage', () => {
  it('renders the chat shell with skip link, main region, and toolbar', async () => {
    await renderChat()
    expect(screen.getByText('Skip to messages').getAttribute('href')).toBe('#chat-messages')
    expect(screen.getByLabelText('Chat')).toBeTruthy()
    expect(screen.getByTestId('chat-toolbar')).toBeTruthy()
    expect(screen.getByTestId('chat-provider')).toBeTruthy()
    expect(screen.getAllByTestId('dynamic').length).toBe(6)
  })

  it('opens the sidebar and fetches initial data on mount', async () => {
    await renderChat()
    expect(mockSetConvSidebarOpen).toHaveBeenCalledWith(true)
    expect(mockFetchStats).toHaveBeenCalled()
    expect(mockFetchAdapterStats).toHaveBeenCalled()
    expect(mockModelFetchInitialData).toHaveBeenCalledWith('gpt2')
    expect(mockAgentFetchInitialData).toHaveBeenCalled()
  })

  it('loads model descriptions on mount', async () => {
    await renderChat()
    expect(mockModelList).toHaveBeenCalled()
  })

  it('loads a session from the search params', async () => {
    mockSearchParamsGet.mockReturnValue('session-42')
    await renderChat()
    expect(mockLoadSession).toHaveBeenCalledWith('session-42')
  })

  it('does not load a session when none is present', async () => {
    await renderChat()
    expect(mockLoadSession).not.toHaveBeenCalled()
  })

  it('fetches suggestions when a model is loaded', async () => {
    await renderChat()
    await waitFor(() => { expect(mockGetSuggestions).toHaveBeenCalled() })
    await waitFor(() => { expect(screen.getByText('Tell me a story')).toBeTruthy() })
  })

  it('does not fetch suggestions when no model is loaded', async () => {
    state.health = { model_loaded: false }
    await renderChat()
    expect(mockGetSuggestions).not.toHaveBeenCalled()
  })

  it('renders ChatSettings when showSettings is enabled', async () => {
    state.ui.showSettings = true
    await renderChat()
    expect(screen.getAllByTestId('dynamic').length).toBe(7)
  })

  it('renders the tool panel when toolPanelOpen is enabled', async () => {
    state.ui.toolPanelOpen = true
    await renderChat()
    expect(screen.getAllByTestId('dynamic').length).toBe(7)
  })

  it('opens the conversation search panel on the search-conversations window event', async () => {
    await renderChat()
    await act(async () => { window.dispatchEvent(new Event('search-conversations')) })
    expect(screen.getAllByTestId('dynamic').length).toBe(6)
  })

  it('starts a new chat on the new-chat window event', async () => {
    await renderChat()
    await act(async () => { window.dispatchEvent(new Event('new-chat')) })
    expect(mockNewChat).toHaveBeenCalled()
  })

  it('copies the last assistant response on copy-last-response', async () => {
    state.chat.messages = [{ id: 'm1', role: 'assistant', content: 'the last answer', timestamp: new Date() }]
    await renderChat()
    await act(async () => { window.dispatchEvent(new Event('copy-last-response')) })
    await waitFor(() => { expect(mockClipboardWrite).toHaveBeenCalledWith('the last answer') })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Last response copied', 'info') })
  })

  it('sends a message from the chat input', async () => {
    await renderChat()
    const textarea = screen.getByLabelText('Chat input')
    await act(async () => { fireEvent.change(textarea, { target: { value: 'hello' } }) })
    await act(async () => { screen.getByTestId('send-btn').click() })
    expect(mockSendMessage).toHaveBeenCalled()
  })

  it('toasts an upload prompt when sending in read mode without a file', async () => {
    state.mode.chatMode = 'read'
    await renderChat()
    await act(async () => { screen.getByTestId('send-btn').click() })
    expect(mockAddToast).toHaveBeenCalledWith('Upload a file first, then ask your question', 'info')
    expect(mockSendMessage).not.toHaveBeenCalled()
  })

  it('generates an image in create mode', async () => {
    state.mode.chatMode = 'create'
    await renderChat()
    const textarea = screen.getByLabelText('Chat input')
    await act(async () => { fireEvent.change(textarea, { target: { value: 'a sunset' } }) })
    await act(async () => { screen.getByTestId('send-btn').click() })
    await waitFor(() => { expect(mockImageGenerate).toHaveBeenCalledWith('a sunset', 'watercolor') })
  })

  it('deletes a message and toasts', async () => {
    state.chat.messages = [{ id: 'm1', role: 'user', content: 'hi', timestamp: new Date() }]
    await renderChat()
    expect(screen.getByTestId('msg-count').textContent).toBe('1')
    await act(async () => { screen.getByTestId('delete-btn').click() })
    expect(screen.getByTestId('msg-count').textContent).toBe('0')
    expect(mockAddToast).toHaveBeenCalledWith('Message deleted', 'info')
  })

  it('adds a bookmark for an existing message', async () => {
    state.chat.messages = [{ id: 'm1', role: 'assistant', content: 'keep this', timestamp: new Date() }]
    await renderChat()
    await act(async () => { screen.getByTestId('bookmark-btn').click() })
    expect(mockAddBookmark).toHaveBeenCalledWith(expect.objectContaining({ id: 'm1' }))
  })

  it('saves message content to the knowledge base', async () => {
    mockKnowledgeAdd.mockResolvedValue({})
    state.chat.messages = [{ id: 'm1', role: 'assistant', content: 'keep this', timestamp: new Date() }]
    await renderChat()
    await act(async () => { screen.getByTestId('knowledge-btn').click() })
    await waitFor(() => { expect(mockKnowledgeAdd).toHaveBeenCalledWith('some content', 'chat-saved', true) })
    expect(mockAddToast).toHaveBeenCalledWith('Saved to knowledge', 'success')
  })

  it('enables the voice overlay in talk mode', async () => {
    state.mode.chatMode = 'talk'
    await renderChat()
    expect(screen.getAllByTestId('dynamic').length).toBe(7)
  })
})
