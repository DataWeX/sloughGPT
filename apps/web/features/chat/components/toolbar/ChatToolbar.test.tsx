import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  return {
    IconSearch: iconMock('search'),
    IconMenu: iconMock('menu'),
    IconPlus: iconMock('plus'),
    IconChat: iconMock('chat'),
    cn: (...args: unknown[]) => args.filter(Boolean).join(' '),
  
Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}
})

vi.mock('./ChatSearchBar', () => ({
  ChatSearchBar: () => <div data-testid="chat-search-bar" />,
}))

vi.mock('./ModelDropdown', () => ({
  ModelDropdown: () => <div data-testid="model-dropdown" />,
}))

vi.mock('./SoulSelectorDropdown', () => ({
  SoulSelectorDropdown: () => <div data-testid="soul-selector-dropdown" />,
}))

vi.mock('./ChatMoreMenu', () => ({
  ChatMoreMenu: () => <div data-testid="chat-more-menu" />,
}))

import { ChatToolbar } from './ChatToolbar'
import { ChatToolbarProvider } from '@/features/chat/contexts/ChatToolbarContext'
import type { ChatToolbarContextValue } from '@/features/chat/contexts/ChatToolbarContext'

const createContextValue = (overrides: Partial<ChatToolbarContextValue> = {}): ChatToolbarContextValue => ({
  conversations: {
    conversations: [],
    sessionIdRef: { current: 's1' },
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
    descriptions: {},
    downloadProgress: {},
    onSelect: vi.fn(),
    onUnload: vi.fn(),
  },
  soul: {
    souls: [],
    current: null,
    onSelect: vi.fn(),
  },
  knowledge: {
    showing: false,
    count: 0,
    context: '',
    onToggle: vi.fn(),
  },
  agent: {
    agents: [],
    current: null,
    onSelect: vi.fn(),
  },
  localEngine: {
    modelUrl: '',
    useLocal: false,
    loading: false,
    archInfo: null,
    onToggle: vi.fn(),
  },
  actions: {
    onVoiceMode: vi.fn(),
    onToggleTools: vi.fn(),
    onExportMarkdown: vi.fn(),
    onCopyMarkdown: vi.fn(),
    onSaveAsDataset: vi.fn(),
    onSystemPrompt: vi.fn(),
    onSearchConversations: vi.fn(),
    hasMessages: false,
    messageCount: 0,
    bookmarkCount: 0,
  },
  health: {
    status: 'ok',
    summary: 'Model loaded: gpt2',
    modelLoaded: true,
    modelType: 'gpt2',
  },
  sidebar: {
    open: false,
    onToggle: vi.fn(),
    onClose: vi.fn(),
  },
  ...overrides,
})

function renderToolbar(ctxValue?: ChatToolbarContextValue) {
  return render(
    <ChatToolbarProvider value={ctxValue || createContextValue()}>
      <ChatToolbar />
    </ChatToolbarProvider>
  )
}

describe('ChatToolbar', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders all core subcomponents', () => {
    renderToolbar()
    expect(screen.getByTestId('chat-search-bar')).toBeDefined()
    expect(screen.getByTestId('model-dropdown')).toBeDefined()
    expect(screen.getByTestId('soul-selector-dropdown')).toBeDefined()
    expect(screen.getByTestId('chat-more-menu')).toBeDefined()
  })

  it('shows sidebar toggle button on mobile', () => {
    renderToolbar()
    expect(screen.getByLabelText('Toggle conversations')).toBeDefined()
  })

  it('calls sidebar.onToggle when menu button clicked', () => {
    const ctx = createContextValue()
    renderToolbar(ctx)
    fireEvent.click(screen.getByLabelText('Toggle conversations'))
    expect(ctx.sidebar.onToggle).toHaveBeenCalledTimes(1)
  })

  it('shows search toggle button on mobile', () => {
    renderToolbar()
    expect(screen.getByLabelText('Toggle search')).toBeDefined()
  })

  it('toggles mobile search visibility', () => {
    const ctx = createContextValue()
    renderToolbar(ctx)
    fireEvent.click(screen.getByLabelText('Toggle search'))
    expect(ctx.search.setShowMobile).toHaveBeenCalledWith(!ctx.search.showMobile)
  })

  it('renders with different sidebar states', () => {
    renderToolbar(createContextValue({ sidebar: { open: true, onToggle: vi.fn(), onClose: vi.fn() } }))
    expect(screen.getByTestId('chat-more-menu')).toBeDefined()
  })

  it('renders with health status ok', () => {
    renderToolbar(createContextValue({ health: { status: 'ok', summary: 'Ready', modelLoaded: true, modelType: 'gpt2' } }))
    expect(screen.getByTestId('model-dropdown')).toBeDefined()
  })

  it('passes knowledge context to toolbar', () => {
    renderToolbar(createContextValue({ knowledge: { showing: true, count: 5, context: 'test', onToggle: vi.fn() } }))
    expect(screen.getByTestId('chat-more-menu')).toBeDefined()
  })
})
