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

import { MembersList } from './MembersList'

afterEach(() => { cleanup() })

const members = [
  { user_id: 'u1', username: 'alice', email: 'alice@example.com', role: 'admin', joined_at: '2025-01-15' },
  { user_id: 'u2', username: 'bob', email: 'bob@example.com', role: 'member', joined_at: '2025-03-20' },
]

describe('MembersList', () => {
  it('renders title', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('Members')).toBeDefined()
  })

  it('renders search input', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByPlaceholderText('Search...')).toBeDefined()
  })

  it('renders member usernames', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('renders member emails', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('alice@example.com')).toBeDefined()
    expect(screen.getByText('bob@example.com')).toBeDefined()
  })

  it('renders role badges', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('shows empty message when no members', () => {
    render(<MembersList members={[]} onRemove={() => {}} />)
    expect(screen.getByText('No members in this workspace')).toBeDefined()
  })

  it('shows loading skeleton', () => {
    const { container } = render(<MembersList members={[]} loading onRemove={() => {}} />)
    expect(container.querySelector('.animate-pulse')).toBeDefined()
  })
})
