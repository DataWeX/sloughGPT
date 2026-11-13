/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceMemberCard } from './WorkspaceMemberCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="member-input" {...props} />,

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

vi.mock('@/components/icons/NavIcons', () => ({
  IconTrash: (props: any) => <svg data-testid="icon-trash" {...props} />,
}))

describe('WorkspaceMemberCard', () => {
  const defaultProps = {
    workspaceName: 'Test Workspace',
    members: [],
    addMemberId: '',
    addMemberRole: 'member',
    onAddMemberIdChange: vi.fn(),
    onAddMemberRoleChange: vi.fn(),
    onAddMember: vi.fn(),
    onRemoveMember: vi.fn(),
  }

  it('renders the title with workspace name', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('Members — Test Workspace')).toBeDefined()
  })

  it('renders the member input', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByTestId('member-input')).toBeDefined()
  })

  it('renders Add button', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('Add')).toBeDefined()
  })

  it('renders no members message when empty', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('renders members when provided', () => {
    const members = [{ user_id: '1', username: 'alice', email: 'alice@test.com', role: 'admin' }]
    render(<WorkspaceMemberCard {...defaultProps} members={members} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('alice@test.com')).toBeDefined()
    expect(screen.getByText('admin')).toBeDefined()
  })

  it('calls onAddMember when Add clicked', () => {
    const onAddMember = vi.fn()
    render(<WorkspaceMemberCard {...defaultProps} addMemberId="user123" onAddMember={onAddMember} />)
    fireEvent.click(screen.getByText('Add'))
    expect(onAddMember).toHaveBeenCalledOnce()
  })

  it('calls onRemoveMember when trash clicked', () => {
    const onRemoveMember = vi.fn()
    const members = [{ user_id: '1', username: 'bob', email: 'bob@test.com', role: 'viewer' }]
    render(<WorkspaceMemberCard {...defaultProps} members={members} onRemoveMember={onRemoveMember} />)
    fireEvent.click(screen.getAllByTestId('icon-trash')[0])
    expect(onRemoveMember).toHaveBeenCalledWith('1')
  })

  it('shows loading state', () => {
    render(<WorkspaceMemberCard {...defaultProps} loading />)
    expect(screen.getByTestId('card-content').querySelector('.animate-pulse')).toBeDefined()
  })
})
