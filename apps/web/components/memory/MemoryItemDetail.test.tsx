// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

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

import { MemoryItemDetail } from './MemoryItemDetail'

afterEach(() => { cleanup() })

const item = {
  id: 'mem-1',
  topic: 'User preferences',
  content: 'Alice prefers dark mode and uses Vim keybindings.',
  source: 'conversation',
  importance: 0.85,
  timestamp: '2025-06-15T10:30:00Z',
}

describe('MemoryItemDetail', () => {
  it('renders placeholder when no item selected', () => {
    render(<MemoryItemDetail item={null} />)
    expect(screen.getByText('Click a memory item to view details.')).toBeDefined()
  })

  it('renders Select item title when no item', () => {
    render(<MemoryItemDetail item={null} />)
    expect(screen.getByText('Select item')).toBeDefined()
  })

  it('renders item topic as title', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText('User preferences')).toBeDefined()
  })

  it('renders item content', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText(/Alice prefers dark mode/)).toBeDefined()
  })

  it('renders Edit and Delete buttons', () => {
    render(<MemoryItemDetail item={item} />)
    expect(screen.getByText('Edit')).toBeDefined()
    expect(screen.getByText('Delete')).toBeDefined()
  })

  it('shows edit textarea in edit mode', () => {
    render(<MemoryItemDetail item={item} editMode editContent={item.content} />)
    expect(screen.getByLabelText('Edit memory content')).toBeDefined()
  })

  it('shows Save and Cancel in edit mode', () => {
    render(<MemoryItemDetail item={item} editMode editContent={item.content} />)
    expect(screen.getByText('Save')).toBeDefined()
    expect(screen.getByText('Cancel')).toBeDefined()
  })
})
