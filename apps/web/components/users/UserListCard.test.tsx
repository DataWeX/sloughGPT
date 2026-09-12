/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { UserListCard, type User } from './UserListCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} {...props}>{children}</button>
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

afterEach(() => cleanup())

const users: User[] = [
  { id: '1', username: 'alice', email: 'alice@test.com', role: 'admin', status: 'active', display_name: 'Alice', last_login_at: '2025-01-15T00:00:00Z' },
  { id: '2', username: 'bob', email: 'bob@test.com', role: 'user', status: 'inactive', display_name: 'Bob', last_login_at: '' },
]

describe('UserListCard', () => {
  it('renders the card title', () => {
    render(<UserListCard users={[]} onEdit={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('Users')).toBeDefined()
  })

  it('renders user usernames', () => {
    render(<UserListCard users={users} onEdit={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('renders user emails', () => {
    render(<UserListCard users={users} onEdit={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText(/alice@test.com/)).toBeDefined()
    expect(screen.getByText(/bob@test.com/)).toBeDefined()
  })

  it('shows no users message when empty', () => {
    render(<UserListCard users={[]} onEdit={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('No users yet')).toBeDefined()
  })

  it('calls onEdit when edit button is clicked', () => {
    const onEdit = vi.fn()
    render(<UserListCard users={users} onEdit={onEdit} onDelete={vi.fn()} />)
    const editButtons = screen.getAllByText('Edit')
    fireEvent.click(editButtons[0])
    expect(onEdit).toHaveBeenCalledWith(users[0])
  })
})
