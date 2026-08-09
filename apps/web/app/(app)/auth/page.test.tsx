import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import AuthPage from './page'

describe('AuthPage', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('renders page header', async () => {
    render(<AuthPage />)
    expect(screen.getAllByText('Auth').length).toBeGreaterThanOrEqual(1)
  })

  it('renders login form', async () => {
    render(<AuthPage />)
    await screen.findAllByPlaceholderText(/username/i)
    expect(screen.getAllByPlaceholderText(/password/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders Token Info card with no token', async () => {
    render(<AuthPage />)
    await screen.findAllByText(/no token/i)
  })
})
