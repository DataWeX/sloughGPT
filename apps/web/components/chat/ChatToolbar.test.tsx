// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockCtx = {
  conversations: { conversations: [], sessionIdRef: { current: '' }, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
  search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
  model: { availableModels: ['gpt2'], current: 'gpt2', loading: null, generating: false, infoMap: {}, onSelect: vi.fn() },
  soul: { souls: [], current: null, onSelect: vi.fn() },
  knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
  agent: { agents: [], current: null, onSelect: vi.fn() },
  localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
  actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), onCopyMarkdown: vi.fn(), onSaveAsDataset: vi.fn(), hasMessages: false, messageCount: 0 },
  health: { status: 'ok' as const, summary: '', modelLoaded: true, modelType: 'gpt2' },
  sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
}

vi.mock('@/contexts/ChatToolbarContext', () => ({
  useChatToolbarContext: () => mockCtx,
}))

vi.mock('./ConversationsDropdown', () => ({ ConversationsDropdown: () => <div data-testid="conversations-dropdown" /> }))
vi.mock('./ChatSearchBar', () => ({ ChatSearchBar: () => <div data-testid="chat-search-bar" /> }))
vi.mock('./ModelDropdown', () => ({ ModelDropdown: () => <div data-testid="model-dropdown" /> }))
vi.mock('./SoulSelectorDropdown', () => ({ SoulSelectorDropdown: () => <div data-testid="soul-selector" /> }))
vi.mock('./AgentSelectorDropdown', () => ({ AgentSelectorDropdown: () => <div data-testid="agent-selector" /> }))
vi.mock('./LocalEngineToggle', () => ({ LocalEngineToggle: () => <div data-testid="local-engine-toggle" /> }))
vi.mock('./ChatMoreMenu', () => ({ ChatMoreMenu: () => <div data-testid="chat-more-menu" /> }))

import { ChatToolbar } from './ChatToolbar'

describe('ChatToolbar', () => {
  afterEach(cleanup)

  it('renders all child dropdowns', () => {
    render(<ChatToolbar />)
    expect(screen.getByTestId('conversations-dropdown')).toBeDefined()
    expect(screen.getByTestId('chat-search-bar')).toBeDefined()
    expect(screen.getByTestId('model-dropdown')).toBeDefined()
    expect(screen.getByTestId('soul-selector')).toBeDefined()
    expect(screen.getByTestId('agent-selector')).toBeDefined()
    expect(screen.getByTestId('chat-more-menu')).toBeDefined()
  })

  it('shows sidebar toggle on mobile', () => {
    render(<ChatToolbar />)
    expect(screen.getByLabelText('Toggle conversations')).toBeDefined()
  })

  it('shows search toggle on mobile', () => {
    render(<ChatToolbar />)
    expect(screen.getByLabelText('Toggle search')).toBeDefined()
  })

  it('shows message count when > 0', () => {
    mockCtx.actions.messageCount = 5
    render(<ChatToolbar />)
    expect(screen.getByText('5')).toBeDefined()
    mockCtx.actions.messageCount = 0
  })

  it('shows knowledge count when > 0', () => {
    mockCtx.knowledge.count = 3
    mockCtx.knowledge.context = '- fact 1\n- fact 2'
    render(<ChatToolbar />)
    expect(screen.getByLabelText('3 knowledge facts active')).toBeDefined()
    mockCtx.knowledge.count = 0
    mockCtx.knowledge.context = ''
  })

  it('shows knowledge popover when showing', () => {
    mockCtx.knowledge.showing = true
    mockCtx.knowledge.count = 2
    mockCtx.knowledge.context = '- Relevant fact\n- Another fact'
    render(<ChatToolbar />)
    expect(screen.getByText(/Injected knowledge/)).toBeDefined()
    mockCtx.knowledge.showing = false
    mockCtx.knowledge.count = 0
    mockCtx.knowledge.context = ''
  })

  it('renders LocalEngineToggle', () => {
    mockCtx.localEngine.modelUrl = 'http://localhost:8080'
    render(<ChatToolbar />)
    expect(screen.getByTestId('local-engine-toggle')).toBeDefined()
    mockCtx.localEngine.modelUrl = ''
  })

  it('calls sidebar.onToggle when sidebar button clicked', () => {
    render(<ChatToolbar />)
    fireEvent.click(screen.getByLabelText('Toggle conversations'))
    expect(mockCtx.sidebar.onToggle).toHaveBeenCalled()
  })
})
