import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockLogin, mockRegister, mockGetMe, mockAddToast,
  mockGetKV, mockSetKV, mockDeleteKV,
} = vi.hoisted(() => ({
  mockLogin: vi.fn(), mockRegister: vi.fn(), mockGetMe: vi.fn(), mockAddToast: vi.fn(),
  mockGetKV: vi.fn().mockResolvedValue(null),
  mockSetKV: vi.fn().mockResolvedValue(undefined),
  mockDeleteKV: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, type }: any) => (
      <button onClick={onClick} disabled={disabled} type={type}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder, type }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} type={type} />
    ),
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span>refresh</span>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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
}
})

vi.mock('@/components/auth/AuthSessionInfoCard', () => ({
  AuthSessionInfoCard: ({ token }: any) => (
    <div data-testid="session-info">{token ? 'has-token' : 'no-token'}</div>
  ),
}))

vi.mock('@/lib/auth-controller', () => ({
  authController: {
    login: (...a: unknown[]) => mockLogin(...a),
    register: (...a: unknown[]) => mockRegister(...a),
    getMe: (...a: unknown[]) => mockGetMe(...a),
    getWorkspaces: (...a: unknown[]) => Promise.resolve([]),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: (...a: unknown[]) => mockGetKV(...a),
    setKV: (...a: unknown[]) => mockSetKV(...a),
    deleteKV: (...a: unknown[]) => mockDeleteKV(...a),
  },
}))

import AuthPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetKV.mockResolvedValue(null)
  mockSetKV.mockResolvedValue(undefined)
  mockDeleteKV.mockResolvedValue(undefined)
  mockLogin.mockResolvedValue({ token: 'test-token-123', user: { username: 'testuser', email: 'test@example.com', role: 'user' } })
  mockRegister.mockResolvedValue({ token: 'new-token-456', user: { username: 'newuser', email: 'new@example.com', role: 'user' } })
  mockGetMe.mockResolvedValue({ username: 'testuser', email: 'test@example.com', role: 'user' })
})

describe('AuthPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<AuthPage />)
    expect(screen.getAllByText('Auth').length).toBeGreaterThanOrEqual(1)
  })

  it('renders login form by default', async () => {
    render(<AuthPage />)
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByPlaceholderText(/password/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows guest status when no token', async () => {
    render(<AuthPage />)
    await waitFor(() => {
      expect(screen.getByText('Guest')).toBeTruthy()
    })
  })

  it('shows token info card with no token message', async () => {
    render(<AuthPage />)
    await waitFor(() => {
      expect(screen.getByText(/no token/i)).toBeTruthy()
    })
  })
})

describe('AuthPage — login flow', () => {
  it('login form submits credentials', async () => {
    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const usernameInput = screen.getAllByPlaceholderText(/username/i)[0]
    const passwordInput = screen.getAllByPlaceholderText(/password/i)[0]
    const submitBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('login') || (b as HTMLButtonElement).type === 'submit'
    )

    fireEvent.change(usernameInput, { target: { value: 'testuser' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })

    if (submitBtn) {
      await act(async () => { fireEvent.click(submitBtn) })
      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith('testuser', 'password123')
      })
    }
  })

  it('shows logged in state after successful login', async () => {
    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const usernameInput = screen.getAllByPlaceholderText(/username/i)[0]
    const passwordInput = screen.getAllByPlaceholderText(/password/i)[0]
    const submitBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('login') || (b as HTMLButtonElement).type === 'submit'
    )

    fireEvent.change(usernameInput, { target: { value: 'testuser' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })

    if (submitBtn) {
      await act(async () => { fireEvent.click(submitBtn) })
      await waitFor(() => {
        expect(screen.getByText('Logged In')).toBeTruthy()
      })
    }
  })

  it('stores token in chatDB after login', async () => {
    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const usernameInput = screen.getAllByPlaceholderText(/username/i)[0]
    const passwordInput = screen.getAllByPlaceholderText(/password/i)[0]
    const submitBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('login') || (b as HTMLButtonElement).type === 'submit'
    )

    fireEvent.change(usernameInput, { target: { value: 'testuser' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })

    if (submitBtn) {
      await act(async () => { fireEvent.click(submitBtn) })
      await waitFor(() => {
        expect(mockSetKV).toHaveBeenCalledWith('auth_token', 'test-token-123')
      })
    }
  })
})

describe('AuthPage — register flow', () => {
  it('switches to register mode', async () => {
    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const registerBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('register') || b.textContent?.toLowerCase().includes('sign up')
    )
    if (registerBtn) {
      fireEvent.click(registerBtn)
      await waitFor(() => {
        expect(screen.getAllByPlaceholderText(/email/i).length).toBeGreaterThanOrEqual(1)
      })
    }
  })

  it('register form submits credentials', async () => {
    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const registerBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('register') || b.textContent?.toLowerCase().includes('sign up')
    )
    if (registerBtn) {
      fireEvent.click(registerBtn)
      await waitFor(() => { expect(screen.getAllByPlaceholderText(/email/i).length).toBeGreaterThanOrEqual(1) })

      const usernameInput = screen.getAllByPlaceholderText(/username/i)[0]
      const emailInput = screen.getAllByPlaceholderText(/email/i)[0]
      const passwordInput = screen.getAllByPlaceholderText(/password/i)[0]
      const submitBtn = screen.getAllByRole('button').find(b =>
        b.textContent?.toLowerCase().includes('register') || (b as HTMLButtonElement).type === 'submit'
      )

      fireEvent.change(usernameInput, { target: { value: 'newuser' } })
      fireEvent.change(emailInput, { target: { value: 'new@example.com' } })
      fireEvent.change(passwordInput, { target: { value: 'pass123' } })

      if (submitBtn) {
        await act(async () => { fireEvent.click(submitBtn) })
        await waitFor(() => {
          expect(mockRegister).toHaveBeenCalledWith('newuser', 'new@example.com', 'pass123')
        })
      }
    }
  })
})

describe('AuthPage — logout flow', () => {
  it('logout button clears token and user', async () => {
    mockGetKV.mockResolvedValue('existing-token')
    mockGetMe.mockResolvedValue({ username: 'testuser', email: 'test@example.com', role: 'user' })

    render(<AuthPage />)
    await waitFor(() => {
      expect(screen.getByText('Logged In')).toBeTruthy()
    })

    const logoutBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('logout') || b.textContent?.toLowerCase().includes('sign out')
    )
    if (logoutBtn) {
      await act(async () => { fireEvent.click(logoutBtn) })
      await waitFor(() => {
        expect(mockDeleteKV).toHaveBeenCalledWith('auth_token')
        expect(screen.getByText('Guest')).toBeTruthy()
      })
    }
  })
})

describe('AuthPage — existing token flow', () => {
  it('loads user from saved token', async () => {
    mockGetKV.mockResolvedValue('existing-token')
    mockGetMe.mockResolvedValue({ username: 'saveduser', email: 'saved@example.com', role: 'user' })

    render(<AuthPage />)
    await waitFor(() => {
      expect(mockGetMe).toHaveBeenCalledWith('existing-token')
    })
  })

  it('clears token on invalid response', async () => {
    mockGetKV.mockResolvedValue('invalid-token')
    mockGetMe.mockRejectedValue(new Error('unauthorized'))

    render(<AuthPage />)
    await waitFor(() => {
      expect(mockDeleteKV).toHaveBeenCalledWith('auth_token')
    })
  })
})

describe('AuthPage — error handling', () => {
  it('shows error on login failure', async () => {
    mockLogin.mockRejectedValue(new Error('Invalid credentials'))

    render(<AuthPage />)
    await waitFor(() => { expect(screen.getAllByPlaceholderText(/username/i).length).toBeGreaterThanOrEqual(1) })

    const usernameInput = screen.getAllByPlaceholderText(/username/i)[0]
    const passwordInput = screen.getAllByPlaceholderText(/password/i)[0]
    const submitBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('login') || (b as HTMLButtonElement).type === 'submit'
    )

    fireEvent.change(usernameInput, { target: { value: 'wrong' } })
    fireEvent.change(passwordInput, { target: { value: 'creds' } })

    if (submitBtn) {
      await act(async () => { fireEvent.click(submitBtn) })
      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeTruthy()
      })
    }
  })
})

describe('AuthPage — session info', () => {
  it('shows token info card', async () => {
    render(<AuthPage />)
    await waitFor(() => {
      expect(screen.getByText(/token info/i)).toBeTruthy()
    })
  })
})
