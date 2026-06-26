// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
vi.mock('@/components/ui/dropdown-menu', () => {
  function DM({ children }: any) { return <div>{children}</div> }
  function DMT({ children, asChild }: any) { return asChild ? <>{children}</> : <button>{children}</button> }
  function DMI({ children, onSelect, disabled }: any) {
    return <button role="menuitem" disabled={disabled} onClick={onSelect}>{children}</button>
  }
  return {
    DropdownMenu: DM, DropdownMenuTrigger: DMT, DropdownMenuContent: ({ children }: any) => <div>{children}</div>,
    DropdownMenuItem: DMI, DropdownMenuSeparator: () => <hr />, DropdownMenuLabel: ({ children }: any) => <div>{children}</div>,
    DropdownMenuCheckboxItem: DMI, DropdownMenuPortal: ({ children }: any) => <div>{children}</div>,
    DropdownMenuGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSub: ({ children }: any) => <div>{children}</div>, DropdownMenuRadioGroup: ({ children }: any) => <div>{children}</div>,
    DropdownMenuSubTrigger: DMT, DropdownMenuSubContent: ({ children }: any) => <div>{children}</div>,
  }
})

import { AgentSelectorDropdown } from './AgentSelectorDropdown'
import type { AgentDef } from '@/lib/agents'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'

const agents: AgentDef[] = [
  { id: 'researcher', name: 'Researcher', description: 'Deep research agent', instructions: 'Research thoroughly.' },
  { id: 'writer', name: 'Writer', description: 'Creative writer', instructions: 'Write creatively.' },
]

function makeCtx(overrides: { agents?: AgentDef[]; current?: AgentDef | null; onSelect?: ReturnType<typeof vi.fn> } = {}): ChatToolbarContextValue {
  const onSelect = overrides.onSelect ?? vi.fn()
  return {
    conversations: { conversations: [], sessionIdRef: { current: '' } as React.MutableRefObject<string>, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
    search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: [], current: null, onSelect: vi.fn() },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: overrides.agents ?? agents, current: overrides.current ?? null, onSelect },
    localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
    actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), onSystemPrompt: vi.fn(), hasMessages: false, messageCount: 0 },
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Parameters<typeof makeCtx>[0] = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><AgentSelectorDropdown /></ChatToolbarProvider>)
}

describe('AgentSelectorDropdown', () => {
  afterEach(cleanup)

  it('renders trigger with Role label when no agent selected', () => {
    renderWithCtx({ current: null })
    expect(screen.getByText('Role')).toBeDefined()
  })

  it('renders trigger with current agent name', () => {
    renderWithCtx({ current: agents[0] })
    const matches = screen.getAllByText('Researcher')
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('shows agent list with descriptions when dropdown opened', () => {
    renderWithCtx({ current: null })
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByText('Deep research agent')).toBeDefined()
    expect(screen.getByText('Creative writer')).toBeDefined()
  })

  it('shows check mark for current agent', () => {
    renderWithCtx({ current: agents[0] })
    fireEvent.click(screen.getByRole('button'))
    const checkSvg = document.querySelector('.text-primary.shrink-0')
    expect(checkSvg).toBeDefined()
  })

  it('calls onSelect when agent clicked', () => {
    const onSelect = vi.fn()
    renderWithCtx({ current: null, onSelect })
    fireEvent.click(screen.getByRole('button'))
    fireEvent.click(screen.getByText('Writer'))
    expect(onSelect).toHaveBeenCalledWith(agents[1])
  })
})
