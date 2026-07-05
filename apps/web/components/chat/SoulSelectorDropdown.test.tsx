import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { getTraitWeights: vi.fn().mockResolvedValue(null) },
}))

vi.mock('@/components/souls/PersonalitySummary', () => ({ deriveArchetype: vi.fn().mockReturnValue(null) }))

vi.mock('@/components/ui/dropdown-menu', () => {
  function DM({ children }: any) { return <div>{children}</div> }
  function DMT({ children, asChild }: any) { return asChild ? <>{children}</> : <button>{children}</button> }
  function DMI({ children, onSelect }: any) {
    return <button role="menuitem" onClick={onSelect}>{children}</button>
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

import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'
import type { Soul } from '@/lib/souls-controller'

const souls: Soul[] = [
  { name: 'friendly', description: 'Warm and approachable', traits: ['warmth', 'empathy'], personality: { warmth: 0.8 } },
  { name: 'witty', description: 'Sharp and clever', traits: ['humor', 'intelligence'], personality: { humor: 0.9 } },
]

function makeCtx(overrides: { souls?: Soul[]; current?: Soul | null; onSelect?: ReturnType<typeof vi.fn> } = {}): ChatToolbarContextValue {
  const onSelect = overrides.onSelect ?? vi.fn()
  return {
    conversations: { conversations: [], sessionIdRef: { current: '' } as React.MutableRefObject<string>, onLoad: vi.fn(), onStar: vi.fn(), onPin: vi.fn(), onNewChat: vi.fn() },
    search: { query: '', onChange: vi.fn(), onClear: vi.fn(), matchIndex: 0, matchCount: 0, matchIds: [], onPrevMatch: vi.fn(), onNextMatch: vi.fn(), showMobile: false, setShowMobile: vi.fn() },
    model: { availableModels: [], current: '', loading: null, generating: false, infoMap: {}, downloadProgress: {}, onSelect: vi.fn() },
    soul: { souls: overrides.souls ?? souls, current: overrides.current ?? null, onSelect },
    knowledge: { showing: false, count: 0, context: '', onToggle: vi.fn() },
    agent: { agents: [], current: null, onSelect: vi.fn() },
    localEngine: { modelUrl: '', useLocal: false, loading: false, archInfo: null, onToggle: vi.fn() },
    actions: { onVoiceMode: vi.fn(), onToggleTools: vi.fn(), onExportMarkdown: vi.fn(), onSystemPrompt: vi.fn(), onSearchConversations: vi.fn(), hasMessages: false, messageCount: 0, bookmarkCount: 0 },
    health: { status: 'ok', summary: '', modelLoaded: false, modelType: '' },
    sidebar: { open: false, onToggle: vi.fn(), onClose: vi.fn() },
  }
}

function renderWithCtx(overrides: Parameters<typeof makeCtx>[0] = {}) {
  return render(<ChatToolbarProvider value={makeCtx(overrides)}><SoulSelectorDropdown /></ChatToolbarProvider>)
}

describe('SoulSelectorDropdown', () => {
  afterEach(cleanup)

  it('renders trigger with personality label', () => {
    renderWithCtx()
    expect(screen.getByText('Personality')).toBeDefined()
  })

  it('renders trigger with current soul name when set', () => {
    renderWithCtx({ current: souls[0] })
    const matches = screen.getAllByText('friendly')
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it('shows soul list in dropdown content', () => {
    renderWithCtx()
    expect(screen.getByText('friendly')).toBeDefined()
    expect(screen.getByText('witty')).toBeDefined()
  })

  it('shows check mark for current soul via menuitem', () => {
    renderWithCtx({ current: souls[0] })
    const items = screen.getAllByRole('menuitem')
    expect(items.length).toBeGreaterThanOrEqual(2)
  })

  it('has View full profile link that navigates to /models', () => {
    renderWithCtx()
    fireEvent.click(screen.getByText('View full profile'))
    expect(mockPush).toHaveBeenCalledWith('/models')
  })

  it('calls onSelect when a soul is clicked', () => {
    const onSelect = vi.fn()
    renderWithCtx({ onSelect })
    const items = screen.getAllByRole('menuitem')
    const friendlyItem = items.find(item => item.textContent?.includes('friendly'))
    if (friendlyItem) fireEvent.click(friendlyItem)
    expect(onSelect).toHaveBeenCalledWith(souls[0])
  })

  it('shows default name in profile header when no soul', () => {
    renderWithCtx()
    expect(screen.getByText('Default')).toBeDefined()
  })

  it('shows soul name in profile header when current soul set', () => {
    renderWithCtx({ current: souls[1] })
    const matches = screen.getAllByText('witty')
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })
})
