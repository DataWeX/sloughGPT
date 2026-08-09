// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { AuthSessionInfoCard } from './AuthSessionInfoCard'

afterEach(() => { cleanup() })

const user = { id: '1', username: 'testuser', email: 'test@example.com' }

function makeToken(expMinutes = 60): string {
  const header = btoa(JSON.stringify({ alg: 'HS256' }))
  const payload = btoa(JSON.stringify({
    sub: '1',
    exp: Math.floor(Date.now() / 1000) + expMinutes * 60,
    iat: Math.floor(Date.now() / 1000) - 300,
  }))
  return `${header}.${payload}.sig`
}

describe('AuthSessionInfoCard', () => {
  it('returns null for no token', () => {
    const { container } = render(<AuthSessionInfoCard token={null} user={user} onLogout={vi.fn()} />)
    expect(container.querySelector('[data-testid="auth-session-info"]')).toBeNull()
  })

  it('returns null for no user', () => {
    const { container } = render(<AuthSessionInfoCard token="x.y.z" user={null} onLogout={vi.fn()} />)
    expect(container.querySelector('[data-testid="auth-session-info"]')).toBeNull()
  })

  it('renders session card', () => {
    render(<AuthSessionInfoCard token={makeToken()} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByTestId('auth-session-info').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Session').length).toBeGreaterThanOrEqual(1)
  })

  it('shows username and email', () => {
    render(<AuthSessionInfoCard token={makeToken()} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByText('testuser').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('test@example.com').length).toBeGreaterThanOrEqual(1)
  })

  it('shows token expiry', () => {
    render(<AuthSessionInfoCard token={makeToken(120)} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByText(/left/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows sign out button', () => {
    render(<AuthSessionInfoCard token={makeToken()} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByText('Sign out').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLogout when sign out clicked', () => {
    const onLogout = vi.fn()
    render(<AuthSessionInfoCard token={makeToken()} user={user} onLogout={onLogout} />)
    fireEvent.click(screen.getAllByText('Sign out')[0])
    expect(onLogout).toHaveBeenCalledOnce()
  })

  it('shows issued date', () => {
    render(<AuthSessionInfoCard token={makeToken()} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByText(/Issued:/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows expired for expired token', () => {
    render(<AuthSessionInfoCard token={makeToken(-10)} user={user} onLogout={vi.fn()} />)
    expect(screen.getAllByText('expired').length).toBeGreaterThanOrEqual(1)
  })
})
