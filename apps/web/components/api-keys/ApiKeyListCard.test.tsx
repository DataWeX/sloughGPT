// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}))

import { ApiKeyListCard } from './ApiKeyListCard'

afterEach(() => cleanup())

const mockKeys = [
  { id: 'k-1', name: 'Prod Key', key_hash: 'sk-abc...xyz', scopes: ['*'], created_at: Date.now() / 1000, revoked: false },
  { id: 'k-2', name: 'Dev Key', key_hash: 'sk-def...uvw', scopes: ['read', 'write'], created_at: Date.now() / 1000, expires_at: Date.now() / 1000 + 86400, revoked: false },
  { id: 'k-3', name: 'Old Key', key_hash: 'sk-old...key', scopes: [], created_at: Date.now() / 1000 - 864000, revoked: true },
]

describe('ApiKeyListCard', () => {
  it('shows empty state', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('No API keys.')).toBeTruthy()
  })

  it('shows active keys', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('(2 active)')).toBeTruthy()
    expect(screen.getByText('Prod Key')).toBeTruthy()
    expect(screen.getByText('Dev Key')).toBeTruthy()
  })

  it('shows revoked keys', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('Old Key')).toBeTruthy()
    const badges = screen.getAllByText('Revoked')
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })

  it('shows key hashes', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('sk-abc...xyz')).toBeTruthy()
    expect(screen.getByText('sk-def...uvw')).toBeTruthy()
  })

  it('shows scopes', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('scopes: read, write')).toBeTruthy()
  })

  it('calls onRotate', () => {
    const onRotate = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRotate={onRotate} />)
    const rotateBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Rotate')
    fireEvent.click(rotateBtns[0])
    expect(onRotate).toHaveBeenCalledWith('k-1')
  })

  it('shows revoke confirmation', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRevoke={onRevoke} />)
    const revokeBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Revoke')
    fireEvent.click(revokeBtns[0])
    expect(screen.getByText('Confirm')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('calls onRevoke after confirm', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRevoke={onRevoke} />)
    const revokeBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Revoke')
    fireEvent.click(revokeBtns[0])
    fireEvent.click(screen.getByText('Confirm'))
    expect(onRevoke).toHaveBeenCalledWith('k-1')
  })
})
