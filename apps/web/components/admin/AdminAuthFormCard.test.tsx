// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,

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

import { AdminAuthFormCard } from './AdminAuthFormCard'

afterEach(() => { cleanup() })

const defaultProps = {
  mode: 'login' as const,
  username: '',
  email: '',
  password: '',
  loading: false,
  error: null,
  onModeChange: vi.fn(),
  onUsernameChange: vi.fn(),
  onEmailChange: vi.fn(),
  onPasswordChange: vi.fn(),
  onSubmit: vi.fn(),
}

describe('AdminAuthFormCard', () => {
  it('renders login title in login mode', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" />)
    expect(screen.getAllByText('Login').length).toBeGreaterThanOrEqual(1)
  })

  it('renders register title in register mode', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="register" />)
    expect(screen.getAllByText('Register').length).toBeGreaterThanOrEqual(1)
  })

  it('renders username input', () => {
    render(<AdminAuthFormCard {...defaultProps} />)
    expect(screen.getByLabelText('Username')).toBeTruthy()
  })

  it('renders password input', () => {
    render(<AdminAuthFormCard {...defaultProps} />)
    expect(screen.getByLabelText('Password')).toBeTruthy()
  })

  it('does not render email input in login mode', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" />)
    expect(screen.queryByLabelText('Email')).toBeNull()
  })

  it('renders email input in register mode', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="register" />)
    expect(screen.getByLabelText('Email')).toBeTruthy()
  })

  it('renders submit button with login label', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" />)
    expect(screen.getByRole('button', { name: 'Login' })).toBeTruthy()
  })

  it('shows loading text when loading', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" loading={true} />)
    expect(screen.getByText('Logging in...')).toBeTruthy()
  })

  it('shows loading text for register', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="register" loading={true} />)
    expect(screen.getByText('Registering...')).toBeTruthy()
  })

  it('shows error message when error provided', () => {
    render(<AdminAuthFormCard {...defaultProps} error="Invalid credentials" />)
    expect(screen.getByText('Invalid credentials')).toBeTruthy()
    expect(screen.getByTestId('auth-error')).toBeTruthy()
  })

  it('hides error when null', () => {
    render(<AdminAuthFormCard {...defaultProps} error={null} />)
    expect(screen.queryByTestId('auth-error')).toBeNull()
  })

  it('shows mode toggle link for login', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" />)
    expect(screen.getByText("Don't have an account?")).toBeTruthy()
    expect(screen.getAllByText('Register').length).toBeGreaterThanOrEqual(1)
  })

  it('shows mode toggle link for register', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="register" />)
    expect(screen.getByText('Already have an account?')).toBeTruthy()
  })

  it('calls onModeChange when toggle clicked', () => {
    const onModeChange = vi.fn()
    render(<AdminAuthFormCard {...defaultProps} mode="login" onModeChange={onModeChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'Register' }))
    expect(onModeChange).toHaveBeenCalledOnce()
  })

  it('calls onSubmit when submit button clicked', () => {
    const onSubmit = vi.fn()
    render(<AdminAuthFormCard {...defaultProps} mode="login" onSubmit={onSubmit} />)
    fireEvent.click(screen.getByRole('button', { name: 'Login' }))
    expect(onSubmit).toHaveBeenCalledOnce()
  })

  it('disables submit button when loading', () => {
    render(<AdminAuthFormCard {...defaultProps} mode="login" loading={true} />)
    const btn = screen.getByRole('button', { name: 'Logging in...' })
    expect(btn).toHaveProperty('disabled', true)
  })

  it('calls onUsernameChange when username input changes', () => {
    const onUsernameChange = vi.fn()
    render(<AdminAuthFormCard {...defaultProps} mode="login" onUsernameChange={onUsernameChange} />)
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'admin' } })
    expect(onUsernameChange).toHaveBeenCalledWith('admin')
  })

  it('calls onPasswordChange when password input changes', () => {
    const onPasswordChange = vi.fn()
    render(<AdminAuthFormCard {...defaultProps} mode="login" onPasswordChange={onPasswordChange} />)
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } })
    expect(onPasswordChange).toHaveBeenCalledWith('secret')
  })

  it('calls onEmailChange when email input changes in register mode', () => {
    const onEmailChange = vi.fn()
    render(<AdminAuthFormCard {...defaultProps} mode="register" onEmailChange={onEmailChange} />)
    fireEvent.change(screen.getByLabelText('Email'), { target: { value: 'a@b.com' } })
    expect(onEmailChange).toHaveBeenCalledWith('a@b.com')
  })
})
