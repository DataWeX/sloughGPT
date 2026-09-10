/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceCreateCard } from './WorkspaceCreateCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="ws-input" {...props} />,
}))

describe('WorkspaceCreateCard', () => {
  const defaultProps = {
    name: '',
    description: '',
    onNameChange: vi.fn(),
    onDescriptionChange: vi.fn(),
    onCreate: vi.fn(),
  }

  it('renders the title', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getByText('Create Workspace')).toBeDefined()
  })

  it('renders name and description inputs', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getAllByTestId('ws-input')).toHaveLength(2)
  })

  it('renders the Create button', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getByText('Create')).toBeDefined()
  })

  it('calls onCreate when Create is clicked', () => {
    const onCreate = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} name="My WS" onCreate={onCreate} />)
    fireEvent.click(screen.getByText('Create'))
    expect(onCreate).toHaveBeenCalledOnce()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} onNameChange={onNameChange} />)
    fireEvent.change(screen.getAllByTestId('ws-input')[0], { target: { value: 'New WS' } })
    expect(onNameChange).toHaveBeenCalledWith('New WS')
  })

  it('calls onDescriptionChange when description input changes', () => {
    const onDescriptionChange = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} onDescriptionChange={onDescriptionChange} />)
    fireEvent.change(screen.getAllByTestId('ws-input')[1], { target: { value: 'A desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('A desc')
  })

  it('disables Create button when name is empty', () => {
    render(<WorkspaceCreateCard {...defaultProps} name="" />)
    expect(screen.getByText('Create')).toHaveProperty('disabled', true)
  })

  it('shows creating state', () => {
    render(<WorkspaceCreateCard {...defaultProps} name="Test" creating />)
    expect(screen.getByText('Creating...')).toBeDefined()
  })
})
