/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { MemberPermissionsList } from './MemberPermissionsList'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const mockMembers = [
  { user_id: '1', username: 'alice', role: 'admin', permissions: ['model.create', 'user.view'] },
  { user_id: '2', username: 'bob', role: 'user', permissions: ['model.view'] },
]

describe('MemberPermissionsList', () => {
  it('renders title with member count', () => {
    render(<MemberPermissionsList members={mockMembers} />)
    expect(screen.getByText('Member Permissions (2)')).toBeDefined()
  })

  it('shows no members message when empty', () => {
    render(<MemberPermissionsList members={[]} />)
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('renders member usernames', () => {
    render(<MemberPermissionsList members={mockMembers} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('renders member roles', () => {
    render(<MemberPermissionsList members={mockMembers} />)
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('user')).toBeDefined()
  })

  it('expands permissions when member is clicked', () => {
    render(<MemberPermissionsList members={mockMembers} />)
    fireEvent.click(screen.getByText('alice'))
    expect(screen.getByText('model.create')).toBeDefined()
    expect(screen.getByText('user.view')).toBeDefined()
  })

  it('collapses expanded permissions on second click', () => {
    render(<MemberPermissionsList members={mockMembers} />)
    fireEvent.click(screen.getByText('alice'))
    fireEvent.click(screen.getByText('alice'))
    expect(screen.queryByText('alice permissions:')).toBeNull()
  })
})
