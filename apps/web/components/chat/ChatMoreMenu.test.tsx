// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
vi.mock('@/components/ui/dropdown-menu', () => {
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
    DropdownMenu: DM, DropdownMenuTrigger: DMT, DropdownMenuContent: ({ children }: any) => <div data-testid="dm-content">{children}</div>,
    DropdownMenuItem: DMI, DropdownMenuSeparator: () => <hr />, DropdownMenuLabel: ({ children }: any) => <div>{children}</div>,
    DropdownMenuCheckboxItem: DMI, DropdownMenuRadioItem: DMI, DropdownMenuGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuPortal: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSub: ({ children }: any) => <div>{children}</div>, DropdownMenuRadioGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSubTrigger: DMT, DropdownMenuSubContent: ({ children }: any) => <div>{children}</div>,
  }
})

import { ChatMoreMenu } from './ChatMoreMenu'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'

function makeCtx(overrides: Partial<ChatToolbarContextValue['actions']> = {}): ChatToolbarContextValue {
  const actions = {
    onVoiceMode: vi.fn(),
    onToggleTools: vi.fn(),
    onExportMarkdown: vi.fn(),
    onCopyMarkdown: undefined,
    onSaveAsDataset: undefined,
    hasMessages: true,
    messageCount: 0,
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

  it('shows Copy to clipboard when onCopyMarkdown provided', () => {
    renderWithCtx({ onCopyMarkdown: vi.fn() })
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Copy to clipboard')).toBeDefined()
  })

  it('does not show Copy to clipboard when onCopyMarkdown not provided', () => {
    renderWithCtx()
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.queryByText('Copy to clipboard')).toBeNull()
  })

  it('shows Save as dataset when onSaveAsDataset provided', () => {
    renderWithCtx({ onSaveAsDataset: vi.fn() })
    fireEvent.click(screen.getByLabelText('More options'))
    expect(screen.getByText('Save as dataset')).toBeDefined()
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
})
