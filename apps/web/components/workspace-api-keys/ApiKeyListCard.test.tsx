/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { ApiKeyListCard } from './ApiKeyListCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconRefresh: (props: any) => <svg data-testid="icon-refresh" {...props} />,
  IconTrash: (props: any) => <svg data-testid="icon-trash" {...props} />,
}))

describe('ApiKeyListCard', () => {
  const mockKeys = [
    { id: '1', name: 'Test Key', key_hash: 'sk_abc123', scopes: ['*'], created_at: 1700000000, revoked: false },
    { id: '2', name: 'Old Key', key_hash: 'sk_xyz789', scopes: ['read'], created_at: 1690000000, revoked: true },
  ]

  it('renders the title', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('Active Keys')).toBeDefined()
  })

  it('renders custom title', () => {
    render(<ApiKeyListCard keys={[]} title="Revoked Keys" />)
    expect(screen.getByText('Revoked Keys')).toBeDefined()
  })

  it('renders empty message when no keys', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('No API keys.')).toBeDefined()
  })

  it('renders key name when keys provided', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('Test Key')).toBeDefined()
  })

  it('renders key hash', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('sk_abc123')).toBeDefined()
  })

  it('calls onRotate when rotate button clicked', () => {
    const onRotate = vi.fn()
    render(<ApiKeyListCard keys={[mockKeys[0]]} onRotate={onRotate} />)
    fireEvent.click(screen.getByTitle('Rotate key'))
    expect(onRotate).toHaveBeenCalledWith('1')
  })

  it('calls onRevoke when revoke button clicked', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={[mockKeys[0]]} onRevoke={onRevoke} />)
    fireEvent.click(screen.getByTitle('Revoke key'))
    expect(onRevoke).toHaveBeenCalledWith('1')
  })

  it('shows Revoked badge for revoked keys', () => {
    render(<ApiKeyListCard keys={[mockKeys[1]]} />)
    expect(screen.getByText('Revoked')).toBeDefined()
  })
})
