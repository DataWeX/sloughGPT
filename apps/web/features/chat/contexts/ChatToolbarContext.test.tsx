import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, renderHook } from '@testing-library/react'
import React from 'react'

import { ChatToolbarProvider, useChatToolbarContext } from './ChatToolbarContext'

const toolbarValue = {
  conversations: {
    conversations: [{ id: 'c1', name: 'Chat one', updated_at: Date.now(), starred: false, pinned: false }] as never,
    sessionIdRef: { current: 'c1' } as never,
    onLoad: vi.fn(),
    onStar: vi.fn(),
    onPin: vi.fn(),
    onNewChat: vi.fn(),
  },
  search: {
    query: '',
    onChange: vi.fn(),
    onClear: vi.fn(),
    matchIndex: 0,
    matchCount: 0,
    matchIds: [],
    onPrevMatch: vi.fn(),
    onNextMatch: vi.fn(),
    showMobile: false,
    setShowMobile: vi.fn(),
  },
  model: {
    availableModels: ['gpt2'],
    current: 'gpt2',
    loading: null,
    generating: false,
    infoMap: {},
    downloadProgress: {},
    onSelect: vi.fn(),
  },
  soul: { souls: [], current: null, onSelect: vi.fn() },
  knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
  agent: { agents: [], current: null, onSelect: vi.fn() },
  localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
  actions: {
    onVoiceMode: vi.fn(),
    onToggleTools: vi.fn(),
    onExportMarkdown: vi.fn(),
    onSystemPrompt: vi.fn(),
    onSearchConversations: vi.fn(),
    hasMessages: false,
    messageCount: 0,
    bookmarkCount: 0,
  },
  health: { status: 'ok' as const, summary: 'healthy', modelLoaded: true, modelType: 'gpt2' },
  sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
}

function renderProvider(value = toolbarValue) {
  return render(
    <ChatToolbarProvider value={value}>
      <div>toolbar child</div>
    </ChatToolbarProvider>
  )
}

function toolbarWrapper({ children }: { children: React.ReactNode }) {
  return <ChatToolbarProvider value={toolbarValue}>{children}</ChatToolbarProvider>
}

afterEach(() => cleanup())

describe('ChatToolbarContext', () => {
  it('renders children inside the provider', () => {
    renderProvider()
    expect(screen.getByText('toolbar child')).toBeDefined()
  })

  it('useChatToolbarContext throws outside provider', () => {
    expect(() => renderHook(() => useChatToolbarContext())).toThrow('useChatToolbarContext must be used within ChatToolbarProvider')
  })

  it('exposes all 9 groups with their values', () => {
    const { result } = renderHook(() => useChatToolbarContext(), { wrapper: toolbarWrapper as any })
    expect(result.current.conversations.conversations[0].name).toBe('Chat one')
    expect(result.current.model.current).toBe('gpt2')
    expect(result.current.health.status).toBe('ok')
    expect(result.current.health.modelLoaded).toBe(true)
    expect(result.current.soul.souls).toEqual([])
    expect(result.current.agent.agents).toEqual([])
    expect(result.current.knowledge.count).toBe(0)
    expect(result.current.sidebar.open).toBe(false)
    expect(result.current.actions.messageCount).toBe(0)
    expect(result.current.localEngine.useLocal).toBe(false)
  })

  it('invokes group callbacks through the hook', () => {
    const { result } = renderHook(() => useChatToolbarContext(), { wrapper: toolbarWrapper as any })
    result.current.actions.onVoiceMode()
    result.current.actions.onToggleTools()
    result.current.conversations.onLoad('x')
    result.current.soul.onSelect({} as never)
    result.current.model.onSelect('qwen')
    result.current.search.onChange('hi')
    result.current.knowledge.onToggle()
    result.current.agent.onSelect({} as never)
    result.current.sidebar.onToggle()
    expect(toolbarValue.actions.onVoiceMode).toHaveBeenCalled()
    expect(toolbarValue.actions.onToggleTools).toHaveBeenCalled()
    expect(toolbarValue.conversations.onLoad).toHaveBeenCalledWith('x')
    expect(toolbarValue.soul.onSelect).toHaveBeenCalled()
    expect(toolbarValue.model.onSelect).toHaveBeenCalledWith('qwen')
    expect(toolbarValue.search.onChange).toHaveBeenCalledWith('hi')
    expect(toolbarValue.knowledge.onToggle).toHaveBeenCalled()
    expect(toolbarValue.agent.onSelect).toHaveBeenCalled()
    expect(toolbarValue.sidebar.onToggle).toHaveBeenCalled()
  })

  it('supports live updates when the value object changes', () => {
    function Harness() {
      const [v, setV] = React.useState(toolbarValue)
      return (
        <ChatToolbarProvider value={v}>
          <button onClick={() => setV({ ...v, health: { ...v.health, modelLoaded: false } })}>toggle</button>
          <Consumer />
        </ChatToolbarProvider>
      )
    }
    function Consumer() {
      const { health } = useChatToolbarContext()
      return <span data-testid="loaded">{String(health.modelLoaded)}</span>
    }
    render(<Harness />)
    expect(screen.getByTestId('loaded').textContent).toBe('true')
    fireEvent.click(screen.getByText('toggle'))
    expect(screen.getByTestId('loaded').textContent).toBe('false')
  })

  it('passes search group values', () => {
    const { result } = renderHook(() => useChatToolbarContext(), { wrapper: toolbarWrapper as any })
    expect(result.current.search.query).toBe('')
    expect(result.current.search.matchIndex).toBe(0)
    expect(result.current.search.matchCount).toBe(0)
    expect(result.current.search.showMobile).toBe(false)
  })
})
