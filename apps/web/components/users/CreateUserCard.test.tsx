/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { CreateUserCard } from './CreateUserCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

afterEach(() => cleanup())

const defaultProps = {
  username: '',
  email: '',
  password: '',
  role: 'user',
  loading: false,
  onUsernameChange: vi.fn(),
  onEmailChange: vi.fn(),
  onPasswordChange: vi.fn(),
  onRoleChange: vi.fn(),
  onCreate: vi.fn(),
}

describe('CreateUserCard', () => {
  it('renders the card title', () => {
    render(<CreateUserCard {...defaultProps} />)
    expect(screen.getByTestId('card-title').textContent).toContain('Create User')
  })

  it('renders username input', () => {
    render(<CreateUserCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Username')).toBeDefined()
  })

  it('renders email input', () => {
    render(<CreateUserCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Email')).toBeDefined()
  })

  it('renders create button', () => {
    render(<CreateUserCard {...defaultProps} />)
    const buttons = screen.getAllByText('Create User')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('calls onCreate when button is clicked', async () => {
    const onCreate = vi.fn()
    render(<CreateUserCard {...defaultProps} username="alice" email="a@b.com" password="secret123" onCreate={onCreate} />)
    const buttons = screen.getAllByText('Create User')
    const btn = buttons[buttons.length - 1]
    await userEvent.click(btn)
    expect(onCreate).toHaveBeenCalledOnce()
  })
})
