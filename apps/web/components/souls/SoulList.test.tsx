// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { SoulList } from './SoulList'

afterEach(() => { cleanup() })

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <span data-testid="card-title" {...props}>{children}</span>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  SearchInput: ({ value, onChange, placeholder, ...props }: any) => (
    <input value={value} onChange={(e: any) => onChange(e.target.value)} placeholder={placeholder} {...props} />
  ),

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
}))

vi.mock('@/lib/time-format', () => ({
  formatShortDate: (d: string) => 'Jan 1, 2025',
}))

import type { Soul } from '@/lib/souls-controller'

const makeSoul = (overrides: Partial<Soul> = {}): Soul => ({
  name: 'test-soul',
  description: 'A test soul',
  traits: ['friendly', 'helpful'],
  personality: { warmth: 0.9, creativity: 0.7 },
  ...overrides,
})

const souls: Soul[] = [
  makeSoul({ name: 'alpha', traits: ['friendly'] }),
  makeSoul({ name: 'beta', traits: ['helpful'], lineage: 'gpt-4' }),
  makeSoul({ name: 'gamma', personality: {} }),
]

describe('SoulList', () => {
  it('renders card with title', () => {
    render(<SoulList souls={[]} currentSoul={null} searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('Personalities').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no souls', () => {
    render(<SoulList souls={[]} currentSoul={null} searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('No personalities found.').length).toBeGreaterThanOrEqual(1)
  })

  it('renders soul list items', () => {
    render(<SoulList souls={souls} currentSoul={null} searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('alpha').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('beta').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('gamma').length).toBeGreaterThanOrEqual(1)
  })

  it('highlights active soul', () => {
    const { container } = render(<SoulList souls={souls} currentSoul="alpha" searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    const badges = container.querySelectorAll('.rounded-full')
    const activeBadges = Array.from(badges).filter(el => el.textContent === 'active')
    expect(activeBadges.length).toBeGreaterThanOrEqual(1)
  })

  it('calls onSwitch when Switch button clicked', () => {
    const onSwitch = vi.fn()
    render(<SoulList souls={souls} currentSoul="alpha" searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={onSwitch} onSelectSoul={vi.fn()} />)
    const switchButtons = screen.getAllByText('Switch')
    fireEvent.click(switchButtons[0])
    expect(onSwitch).toHaveBeenCalledWith('beta')
  })

  it('disables Switch button when switching', () => {
    render(<SoulList souls={souls} currentSoul="alpha" searchQuery="" onSearchChange={vi.fn()} switching="beta" onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('Switching...').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onSelectSoul when soul row clicked', () => {
    const onSelect = vi.fn()
    render(<SoulList souls={souls} currentSoul={null} searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={onSelect} />)
    const rows = screen.getAllByRole('button')
    fireEvent.click(rows[0])
    expect(onSelect).toHaveBeenCalled()
  })

  it('shows traits badges', () => {
    render(<SoulList souls={souls} currentSoul={null} searchQuery="" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('friendly').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('helpful').length).toBeGreaterThanOrEqual(1)
  })

  it('filters souls by search query', () => {
    render(<SoulList souls={souls} currentSoul={null} searchQuery="alpha" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('alpha').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('beta')).toBeNull()
  })

  it('shows search empty state', () => {
    render(<SoulList souls={souls} currentSoul={null} searchQuery="zzz" onSearchChange={vi.fn()} switching={null} onSwitch={vi.fn()} onSelectSoul={vi.fn()} />)
    expect(screen.getAllByText('No personalities match your search.').length).toBeGreaterThanOrEqual(1)
  })
})
