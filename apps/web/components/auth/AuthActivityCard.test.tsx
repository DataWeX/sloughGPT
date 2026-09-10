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
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '5m ago',
}))

import { AuthActivityCard, recordAuthEvent } from './AuthActivityCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('AuthActivityCard', () => {
  it('shows empty state', () => {
    render(<AuthActivityCard />)
    expect(screen.getByText('Activity')).toBeTruthy()
    expect(screen.getByText('No activity recorded.')).toBeTruthy()
  })

  it('shows recorded events', () => {
    recordAuthEvent('login', '192.168.1.1')
    recordAuthEvent('logout')
    render(<AuthActivityCard />)
    expect(screen.getByText('(2)')).toBeTruthy()
    expect(screen.getByText('login')).toBeTruthy()
    expect(screen.getByText('logout')).toBeTruthy()
    expect(screen.getByText('192.168.1.1')).toBeTruthy()
  })

  it('filters by type', () => {
    recordAuthEvent('login')
    recordAuthEvent('register')
    recordAuthEvent('login')
    render(<AuthActivityCard />)
    const loginBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('login'))
    if (loginBtn) fireEvent.click(loginBtn)
    expect(screen.queryByText('register')).toBeNull()
  })

  it('clears activity', () => {
    recordAuthEvent('login')
    recordAuthEvent('logout')
    render(<AuthActivityCard />)
    fireEvent.click(screen.getByText('Clear'))
    expect(screen.getByText('No activity recorded.')).toBeTruthy()
  })

  it('loads from localStorage', () => {
    localStorage.setItem('sloughgpt-auth-activity', JSON.stringify([
      { id: 'evt-1', type: 'failed_login', timestamp: Date.now(), details: 'Wrong password' },
    ]))
    render(<AuthActivityCard />)
    expect(screen.getByText('failed login')).toBeTruthy()
    expect(screen.getByText('Wrong password')).toBeTruthy()
  })
})
