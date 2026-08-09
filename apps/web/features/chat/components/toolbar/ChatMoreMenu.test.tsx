import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
vi.mock('@sloughgpt/strui', () => {
  function DM({ children }: any) { return <div>{children}</div> }
  function DMT({ children, asChild, ...props }: any) {
    const btnProps: any = {}
    if (props.onClick) btnProps.onClick = props.onClick
    return asChild ? <>{children}</> : <button {...btnProps}>{children}</button>
  }
  function DMI({ children, onSelect, disabled, ...props }: any) {
    return <button role="menuitem" aria-disabled={disabled} onClick={onSelect} data-testid="dm-item">{children}</button>
  }
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    DropdownMenu: DM, DropdownMenuTrigger: DMT, DropdownMenuContent: ({ children }: any) => <div data-testid="dm-content">{children}</div>,
    DropdownMenuItem: DMI, DropdownMenuSeparator: () => <hr />, DropdownMenuLabel: ({ children }: any) => <div>{children}</div>,
    DropdownMenuCheckboxItem: DMI, DropdownMenuRadioItem: DMI, DropdownMenuGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuPortal: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSub: ({ children }: any) => <div>{children}</div>, DropdownMenuRadioGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSubTrigger: DMT, DropdownMenuSubContent: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, variant, size, ...rest }: any) => (
      <button onClick={onClick} data-variant={variant} data-size={size} {...rest}>{children}</button>
    ),
    IconMore: () => <span data-testid="icon-more">more</span>,
    IconSettings: () => <span data-testid="icon-settings">settings</span>,
    IconSearch: () => <span data-testid="icon-search">search</span>,
    IconCopy: () => <span data-testid="icon-copy">copy</span>,
    IconExport: () => <span data-testid="icon-export">export</span>,
    IconDocument: () => <span data-testid="icon-document">document</span>,
  }
})

import { ChatMoreMenu } from './ChatMoreMenu'
import { ChatToolbarProvider } from '@/features/chat/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/features/chat/contexts/ChatToolbarContext'

function makeCtx(overrides: Partial<ChatToolbarContextValue['actions']> = {}): ChatToolbarContextValue {
  const actions = {
    onVoiceMode: vi.fn(),
    onToggleTools: vi.fn(),
    onExportMarkdown: vi.fn(),
    onCopyMarkdown: undefined,
    onSaveAsDataset: undefined,
    onSystemPrompt: vi.fn(),
    onSearchConversations: vi.fn(),
    hasMessages: true,
    messageCount: 0,
    bookmarkCount: 0,
    ...overrides,
  }
  return {
    conversations: { conversations: [], sessionIdRef: { current: '' } as React.MutableRefObject<string>, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
    search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: [], current: null, onSelect: vi.fn() },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: [], current: null, onSelect: vi.fn() },
    localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
    actions,
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Partial<ChatToolbarContextValue['actions']> = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><ChatMoreMenu /></ChatToolbarProvider>)
}

describe('ChatMoreMenu', () => {
  afterEach(cleanup)

  it('renders trigger button with More icon', () => {
    renderWithCtx()
    expect(screen.getByLabelText('More options')).toBeDefined()
  })

  it('shows Voice Mode menu item', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Voice Mode')).toBeDefined()
  })

  it('shows Tools Panel menu item', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Tools Panel')).toBeDefined()
  })

  it('shows Export Markdown menu item', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Export Markdown')).toBeDefined()
  })

  it('shows Copy when onCopyMarkdown provided', () => {
    renderWithCtx({ onCopyMarkdown: vi.fn() })
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Copy')).toBeDefined()
  })

  it('does not show Copy when onCopyMarkdown not provided', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.queryByText('Copy')).toBeNull()
  })

  it('shows Save as Dataset when onSaveAsDataset provided', () => {
    renderWithCtx({ onSaveAsDataset: vi.fn() })
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Save as Dataset')).toBeDefined()
  })

  it('disables Export Markdown when hasMessages is false', () => {
    renderWithCtx({ hasMessages: false })
    fireEvent.click(screen.getByLabelText('More options'))
    const items = screen.getAllByText('Export Markdown')
    const parent = items[0].closest('[role="menuitem"]')
    expect(parent?.getAttribute('aria-disabled')).toBe('true')
  })

  it('calls onVoiceMode when Voice Mode selected', () => {
    const onVoiceMode = vi.fn()
    renderWithCtx({ onVoiceMode })
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('Voice Mode'))
    expect(onVoiceMode).toHaveBeenCalled()
  })

  it('calls onToggleTools when Tools Panel selected', () => {
    const onToggleTools = vi.fn()
    renderWithCtx({ onToggleTools })
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('Tools Panel'))
    expect(onToggleTools).toHaveBeenCalled()
  })

  it('shows System Prompt menu item', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('System Prompt')).toBeDefined()
  })

  it('calls onSystemPrompt when System Prompt selected', () => {
    const onSystemPrompt = vi.fn()
    renderWithCtx({ onSystemPrompt })
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('System Prompt'))
    expect(onSystemPrompt).toHaveBeenCalled()
  })

  it('calls onSearchConversations when Search selected', () => {
    const onSearchConversations = vi.fn()
    renderWithCtx({ onSearchConversations })
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('Search'))
    expect(onSearchConversations).toHaveBeenCalled()
  })

  it('renders agent section when agents are present', () => {
    const ctx = makeCtx()
    ctx.agent.agents = [
      { id: 'a1', name: 'Researcher', description: 'Finds info', instructions: '', capabilities: ['search', 'read'] },
    ]
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Researcher')).toBeDefined()
    expect(screen.getByText('Finds info')).toBeDefined()
    expect(screen.getByText('2 capabilities')).toBeDefined()
  })

  it('shows active checkmark for current agent', () => {
    const ctx = makeCtx()
    ctx.agent.agents = [
      { id: 'a1', name: 'Researcher', description: '', instructions: '', capabilities: [] },
      { id: 'a2', name: 'Writer', description: '', instructions: '', capabilities: [] },
    ]
    ctx.agent.current = { id: 'a1', name: 'Researcher', description: '', instructions: '' }
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Agent — Researcher')).toBeDefined()
  })

  it('hides agent section when no agents', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.queryByText('Agent')).toBeNull()
  })

  it('calls agent.onSelect when agent clicked', () => {
    const ctx = makeCtx()
    const onSelect = vi.fn()
    ctx.agent.agents = [{ id: 'a1', name: 'Researcher', description: '', instructions: '', capabilities: [] }]
    ctx.agent.onSelect = onSelect
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('Researcher'))
    expect(onSelect).toHaveBeenCalled()
  })

  it('renders Local Engine toggle', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Local Engine')).toBeDefined()
  })

  it('calls localEngine.onToggle when Local Engine clicked', () => {
    const ctx = makeCtx()
    const onToggle = vi.fn()
    ctx.localEngine.onToggle = onToggle
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    fireEvent.click(screen.getByText('Local Engine'))
    expect(onToggle).toHaveBeenCalled()
  })

  it('shows loading indicator when localEngine.loading is true', () => {
    const ctx = makeCtx()
    ctx.localEngine.loading = true
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    const pulseEl = document.querySelector('.animate-pulse')
    expect(pulseEl).toBeTruthy()
  })

  it('shows checkmark icon when localEngine.useLocal is true', () => {
    const ctx = makeCtx()
    ctx.localEngine.useLocal = true
    render(
      <ChatToolbarProvider value={ctx}>
        <ChatMoreMenu />
      </ChatToolbarProvider>
    )
    fireEvent.click(screen.getByLabelText('More options'))
    const checkEl = document.querySelector('.bg-primary.border-primary')
    expect(checkEl).toBeTruthy()
  })
})
