// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'
import { ChatSearchBar } from './ChatSearchBar'

function makeCtx(overrides: Partial<ChatToolbarContextValue['search']> = {}): ChatToolbarContextValue {
  const search = {
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
    ...overrides,
  }
  return {
    conversations: { conversations: [], sessionIdRef: { current: '' } as React.MutableRefObject<string>, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
    search,
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: [], current: null, onSelect: vi.fn() },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: [], current: null, onSelect: vi.fn() },
    localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
    actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), hasMessages: false, messageCount: 0 },
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Partial<ChatToolbarContextValue['search']> = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><ChatSearchBar /></ChatToolbarProvider>)
}

describe('ChatSearchBar', () => {
  afterEach(cleanup)

  it('renders search input with placeholder', () => {
    renderWithCtx()
    expect(screen.getByPlaceholderText('Search...')).toBeDefined()
  })

  it('has aria-label on input', () => {
    renderWithCtx()
    expect(screen.getByLabelText('Search messages')).toBeDefined()
  })

  it('shows match counter when searchQuery and matchCount > 0', () => {
    renderWithCtx({ query: 'hello', matchIndex: 0, matchCount: 3 })
    expect(screen.getByText('1/3')).toBeDefined()
  })

  it('shows prev/next/clear buttons when searchQuery is set', () => {
    renderWithCtx({ query: 'hello', matchIndex: 0, matchCount: 1 })
    expect(screen.getByLabelText('Previous match')).toBeDefined()
    expect(screen.getByLabelText('Next match')).toBeDefined()
    expect(screen.getByLabelText('Clear search')).toBeDefined()
  })

  it('hides navigation buttons when searchQuery is empty', () => {
    renderWithCtx()
    expect(screen.queryByLabelText('Previous match')).toBeNull()
  })

  it('disables prev/next when matchCount is 0', () => {
    renderWithCtx({ query: 'hello', matchCount: 0 })
    expect(screen.getByLabelText('Previous match')).toHaveProperty('disabled', true)
    expect(screen.getByLabelText('Next match')).toHaveProperty('disabled', true)
  })

  it('calls onSearchChange when typing', () => {
    const onChange = vi.fn()
    renderWithCtx({ onChange })
    fireEvent.change(screen.getByPlaceholderText('Search...'), { target: { value: 'test' } })
    expect(onChange).toHaveBeenCalledWith('test')
  })

  it('calls onClear when clear button clicked', () => {
    const onClear = vi.fn()
    renderWithCtx({ query: 'hello', matchCount: 1, onClear })
    fireEvent.click(screen.getByLabelText('Clear search'))
    expect(onClear).toHaveBeenCalled()
  })

  it('calls onPrevMatch when previous button clicked', () => {
    const onPrev = vi.fn()
    renderWithCtx({ query: 'hello', matchIndex: 1, matchCount: 3, onPrevMatch: onPrev })
    fireEvent.click(screen.getByLabelText('Previous match'))
    expect(onPrev).toHaveBeenCalled()
  })

  it('calls onNextMatch when next button clicked', () => {
    const onNext = vi.fn()
    renderWithCtx({ query: 'hello', matchIndex: 1, matchCount: 3, onNextMatch: onNext })
    fireEvent.click(screen.getByLabelText('Next match'))
    expect(onNext).toHaveBeenCalled()
  })
})
