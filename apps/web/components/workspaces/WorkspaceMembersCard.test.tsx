/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { WorkspaceMembersCard } from './WorkspaceMembersCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,

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

const mockMembers = [
  { user_id: 'u1', username: 'alice', email: 'alice@test.com', role: 'admin' },
  { user_id: 'u2', username: 'bob', email: 'bob@test.com', role: 'member' },
]

describe('WorkspaceMembersCard', () => {
  it('renders the title with workspace name', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="My Workspace"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Members — My Workspace')).toBeDefined()
  })

  it('renders member usernames when provided', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={mockMembers}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('shows empty state when no members', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('renders Add button and role selector', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Add')).toBeDefined()
    expect(screen.getByText('Viewer')).toBeDefined()
    expect(screen.getByText('Member')).toBeDefined()
    expect(screen.getByText('Admin')).toBeDefined()
  })

  it('displays member roles', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={mockMembers}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('shows loading state', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        loading={true}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Members — Test')).toBeDefined()
  })
})
