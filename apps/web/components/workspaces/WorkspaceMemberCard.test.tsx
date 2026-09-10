/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceMemberCard } from './WorkspaceMemberCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="member-input" {...props} />,
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconTrash: (props: any) => <svg data-testid="icon-trash" {...props} />,
}))

describe('WorkspaceMemberCard', () => {
  const defaultProps = {
    workspaceName: 'Test Workspace',
    members: [],
    addMemberId: '',
    addMemberRole: 'member',
    onAddMemberIdChange: vi.fn(),
    onAddMemberRoleChange: vi.fn(),
    onAddMember: vi.fn(),
    onRemoveMember: vi.fn(),
  }

  it('renders the title with workspace name', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('Members — Test Workspace')).toBeDefined()
  })

  it('renders the member input', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByTestId('member-input')).toBeDefined()
  })

  it('renders Add button', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('Add')).toBeDefined()
  })

  it('renders no members message when empty', () => {
    render(<WorkspaceMemberCard {...defaultProps} />)
    expect(screen.getByText('No members')).toBeDefined()
  })

  it('renders members when provided', () => {
    const members = [{ user_id: '1', username: 'alice', email: 'alice@test.com', role: 'admin' }]
    render(<WorkspaceMemberCard {...defaultProps} members={members} />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('alice@test.com')).toBeDefined()
    expect(screen.getByText('admin')).toBeDefined()
  })

  it('calls onAddMember when Add clicked', () => {
    const onAddMember = vi.fn()
    render(<WorkspaceMemberCard {...defaultProps} addMemberId="user123" onAddMember={onAddMember} />)
    fireEvent.click(screen.getByText('Add'))
    expect(onAddMember).toHaveBeenCalledOnce()
  })

  it('calls onRemoveMember when trash clicked', () => {
    const onRemoveMember = vi.fn()
    const members = [{ user_id: '1', username: 'bob', email: 'bob@test.com', role: 'viewer' }]
    render(<WorkspaceMemberCard {...defaultProps} members={members} onRemoveMember={onRemoveMember} />)
    fireEvent.click(screen.getAllByTestId('icon-trash')[0])
    expect(onRemoveMember).toHaveBeenCalledWith('1')
  })

  it('shows loading state', () => {
    render(<WorkspaceMemberCard {...defaultProps} loading />)
    expect(screen.getByTestId('card-content').querySelector('.animate-pulse')).toBeDefined()
  })
})
