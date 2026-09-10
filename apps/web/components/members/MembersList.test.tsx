// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,
}))

import { MembersList } from './MembersList'

afterEach(() => { cleanup() })

const members = [
  { user_id: 'u1', username: 'alice', email: 'alice@example.com', role: 'admin', joined_at: '2025-01-15' },
  { user_id: 'u2', username: 'bob', email: 'bob@example.com', role: 'member', joined_at: '2025-03-20' },
]

describe('MembersList', () => {
  it('renders title', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('Members')).toBeDefined()
  })

  it('renders search input', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByPlaceholderText('Search...')).toBeDefined()
  })

  it('renders member usernames', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('renders member emails', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('alice@example.com')).toBeDefined()
    expect(screen.getByText('bob@example.com')).toBeDefined()
  })

  it('renders role badges', () => {
    render(<MembersList members={members} onRemove={() => {}} />)
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('shows empty message when no members', () => {
    render(<MembersList members={[]} onRemove={() => {}} />)
    expect(screen.getByText('No members in this workspace')).toBeDefined()
  })

  it('shows loading skeleton', () => {
    const { container } = render(<MembersList members={[]} loading onRemove={() => {}} />)
    expect(container.querySelector('.animate-pulse')).toBeDefined()
  })
})
