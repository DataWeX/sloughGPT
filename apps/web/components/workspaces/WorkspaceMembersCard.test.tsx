/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { WorkspaceMembersCard } from './WorkspaceMembersCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,
}))

const mockMembers = [
  { user_id: 'u1', username: 'alice', email: 'alice@test.com', role: 'admin' },
  { user_id: 'u2', username: 'bob', email: 'bob@test.com', role: 'member' },
]

describe('WorkspaceMembersCard', () => {
  it('renders the title with workspace name', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="My Workspace"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Members — My Workspace')).toBeDefined()
  })

  it('renders member usernames when provided', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={mockMembers}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('bob')).toBeDefined()
  })

  it('shows empty state when no members', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('renders Add button and role selector', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Add')).toBeDefined()
    expect(screen.getByText('Viewer')).toBeDefined()
    expect(screen.getByText('Member')).toBeDefined()
    expect(screen.getByText('Admin')).toBeDefined()
  })

  it('displays member roles', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={mockMembers}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('shows loading state', () => {
    render(
      <WorkspaceMembersCard
        workspaceName="Test"
        members={[]}
        loading={true}
        onAddMember={vi.fn()}
        onRemoveMember={vi.fn()}
      />,
    )
    expect(screen.getByText('Members — Test')).toBeDefined()
  })
})
