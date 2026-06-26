// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'
import { LocalEngineToggle } from './LocalEngineToggle'

function makeCtx(overrides: Partial<ChatToolbarContextValue['localEngine']> = {}): ChatToolbarContextValue {
  const localEngine = {
    modelUrl: '',
    useLocal: false,
    loading: false,
    archInfo: null,
    onToggle: vi.fn(),
    ...overrides,
  }
  return {
    conversations: { conversations: [], sessionIdRef: { current: '' } as React.MutableRefObject<string>, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
    search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: [], current: null, onSelect: vi.fn() },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: [], current: null, onSelect: vi.fn() },
    localEngine,
    actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), onSystemPrompt: vi.fn(), hasMessages: false, messageCount: 0 },
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Partial<ChatToolbarContextValue['localEngine']> = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><LocalEngineToggle /></ChatToolbarProvider>)
}

describe('LocalEngineToggle', () => {
  afterEach(cleanup)

  it('returns null when not visible', () => {
    const { container } = renderWithCtx({ modelUrl: '' })
    expect(container.innerHTML).toBe('')
  })

  it('renders Server label when visible and not local', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: false, loading: false, archInfo: null })
    expect(screen.getByText('Server')).toBeDefined()
  })

  it('renders Local label when using local engine', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: true, loading: false, archInfo: null })
    expect(screen.getByText('Local')).toBeDefined()
  })

  it('renders Loading label when loading', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: false, loading: true, archInfo: null })
    expect(screen.getByText('Loading')).toBeDefined()
  })

  it('disables button when loading', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: false, loading: true, archInfo: null })
    const btn = screen.getByRole('button')
    expect(btn.hasAttribute('disabled')).toBe(true)
  })

  it('shows arch info in title when localArchInfo is set', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: true, loading: false, archInfo: 'MPS' })
    const btn = screen.getByRole('button')
    expect(btn.getAttribute('title')).toContain('MPS')
  })

  it('has aria-pressed when useLocalEngine', () => {
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: true, loading: false, archInfo: null })
    const btn = screen.getByRole('button')
    expect(btn.getAttribute('aria-pressed')).toBe('true')
  })

  it('calls onToggle when clicked', () => {
    const onToggle = vi.fn()
    renderWithCtx({ modelUrl: 'http://server:8080', useLocal: false, loading: false, archInfo: null, onToggle })
    fireEvent.click(screen.getByRole('button'))
    expect(onToggle).toHaveBeenCalled()
  })
})
