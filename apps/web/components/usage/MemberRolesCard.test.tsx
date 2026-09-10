/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { MemberRolesCard } from './MemberRolesCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

describe('MemberRolesCard', () => {
  it('renders the card title', () => {
    render(<MemberRolesCard roles={{}} />)
    expect(screen.getByText('Member Roles')).toBeDefined()
  })

  it('renders role names', () => {
    render(<MemberRolesCard roles={{ admin: 2, user: 10 }} />)
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('user')).toBeDefined()
  })

  it('renders role counts', () => {
    render(<MemberRolesCard roles={{ admin: 2, user: 10 }} />)
    expect(screen.getByText('2')).toBeDefined()
    expect(screen.getByText('10')).toBeDefined()
  })

  it('shows empty state when no roles', () => {
    render(<MemberRolesCard roles={{}} />)
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('capitalizes role names', () => {
    render(<MemberRolesCard roles={{ viewer: 5 }} />)
    expect(screen.getByText('viewer')).toBeDefined()
  })
})
