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
