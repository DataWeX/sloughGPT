// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

Object.defineProperty(navigator, 'clipboard', {
  value: { writeText: vi.fn().mockResolvedValue(undefined) },
})

import { AuthTokenCard } from './AuthTokenCard'

afterEach(() => cleanup())

function makeToken(payload: Record<string, unknown>): string {
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }))
  const body = btoa(JSON.stringify(payload))
  return `${header}.${body}.fake-sig`
}

describe('AuthTokenCard', () => {
  it('renders nothing when no token', () => {
    const { container } = render(<AuthTokenCard token={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows token details', () => {
    const token = makeToken({ sub: 'user-1', role: 'admin', exp: Math.floor(Date.now() / 1000) + 3600, iat: Math.floor(Date.now() / 1000) })
    render(<AuthTokenCard token={token} />)
    expect(screen.getByText('Token Details')).toBeTruthy()
    expect(screen.getByText('user-1')).toBeTruthy()
    expect(screen.getByText('admin')).toBeTruthy()
  })

  it('shows copy button', () => {
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) + 3600 })
    render(<AuthTokenCard token={token} />)
    expect(screen.getByText('Copy')).toBeTruthy()
  })

  it('copies token to clipboard', async () => {
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) + 3600 })
    render(<AuthTokenCard token={token} />)
    fireEvent.click(screen.getByText('Copy'))
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(token)
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeTruthy()
    })
  })

  it('shows verify button when onVerify provided', () => {
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) + 3600 })
    render(<AuthTokenCard token={token} onVerify={vi.fn()} />)
    expect(screen.getByText('Verify')).toBeTruthy()
  })

  it('calls onVerify and shows valid result', async () => {
    const onVerify = vi.fn().mockResolvedValue({ valid: true })
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) + 3600 })
    render(<AuthTokenCard token={token} onVerify={onVerify} />)
    fireEvent.click(screen.getByText('Verify'))
    await waitFor(() => {
      expect(screen.getByText(/Token is valid/)).toBeTruthy()
    })
  })

  it('shows expired token warning', () => {
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) - 3600 })
    render(<AuthTokenCard token={token} />)
    expect(screen.getByText('Expired')).toBeTruthy()
  })

  it('shows raw token', () => {
    const token = makeToken({ sub: 'u', exp: Math.floor(Date.now() / 1000) + 3600 })
    render(<AuthTokenCard token={token} />)
    expect(screen.getByText(token)).toBeTruthy()
  })
})
