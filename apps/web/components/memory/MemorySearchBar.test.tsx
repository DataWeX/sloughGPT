// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,

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

import { MemorySearchBar } from './MemorySearchBar'

afterEach(() => { cleanup() })

describe('MemorySearchBar', () => {
  it('renders search input', () => {
    render(<MemorySearchBar query="" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} />)
    expect(screen.getByPlaceholderText('Search memory...')).toBeDefined()
  })

  it('renders Search button', () => {
    render(<MemorySearchBar query="" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('shows loading state when searching', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} searching />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })

  it('renders Clear button when hasResults is true', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} hasResults />)
    expect(screen.getByText('Clear')).toBeDefined()
  })

  it('does not render Clear button when hasResults is false', () => {
    render(<MemorySearchBar query="test" onQueryChange={() => {}} onSearch={() => {}} onClear={() => {}} hasResults={false} />)
    expect(screen.queryByText('Clear')).toBeNull()
  })

  it('calls onSearch when Search button is clicked', () => {
    const onSearch = vi.fn()
    render(<MemorySearchBar query="q" onQueryChange={() => {}} onSearch={onSearch} onClear={() => {}} />)
    screen.getByText('Search').click()
    expect(onSearch).toHaveBeenCalled()
  })
})
