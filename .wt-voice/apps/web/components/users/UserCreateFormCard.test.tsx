/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { UserCreateFormCard } from './UserCreateFormCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ placeholder, ...props }: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input data-testid={`input-${placeholder}`} placeholder={placeholder} {...props} />
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

const defaultProps = {
  username: '',
  email: '',
  password: '',
  role: 'user',
  creating: false,
  onUsernameChange: vi.fn(),
  onEmailChange: vi.fn(),
  onPasswordChange: vi.fn(),
  onRoleChange: vi.fn(),
  onCreate: vi.fn(),
}

describe('UserCreateFormCard', () => {
  it('renders the card title', () => {
    render(<UserCreateFormCard {...defaultProps} />)
    expect(screen.getByText('Create User', { selector: 'h3' })).toBeDefined()
  })

  it('renders input fields', () => {
    render(<UserCreateFormCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Username')).toBeDefined()
    expect(screen.getByPlaceholderText('Email')).toBeDefined()
    expect(screen.getByPlaceholderText('Password (min 8 chars)')).toBeDefined()
  })

  it('renders create button', () => {
    render(<UserCreateFormCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: 'Create User' })).toBeDefined()
  })

  it('calls onCreate when button is clicked', () => {
    render(<UserCreateFormCard {...defaultProps} username="test" email="t@t.com" password="pass1234" />)
    fireEvent.click(screen.getByRole('button', { name: 'Create User' }))
    expect(defaultProps.onCreate).toHaveBeenCalledOnce()
  })

  it('shows Creating... when creating', () => {
    render(<UserCreateFormCard {...defaultProps} creating />)
    expect(screen.getByText('Creating...')).toBeDefined()
  })
})
