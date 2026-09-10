/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceGeneralCard } from './WorkspaceGeneralCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Input: (props: any) => <input data-testid="input" {...props} />,
}))

describe('WorkspaceGeneralCard', () => {
  it('renders the title', () => {
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('General')).toBeDefined()
  })

  it('renders Name label and input', () => {
    render(
      <WorkspaceGeneralCard name="Test" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('Name')).toBeDefined()
    expect(screen.getByDisplayValue('Test')).toBeDefined()
  })

  it('renders Description label and textarea', () => {
    render(
      <WorkspaceGeneralCard name="" description="My desc" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('Description')).toBeDefined()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={onNameChange} onDescriptionChange={vi.fn()} />,
    )
    const inputs = screen.getAllByTestId('input')
    fireEvent.change(inputs[0], { target: { value: 'New Name' } })
    expect(onNameChange).toHaveBeenCalledWith('New Name')
  })

  it('calls onDescriptionChange when textarea changes', () => {
    const onDescriptionChange = vi.fn()
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={vi.fn()} onDescriptionChange={onDescriptionChange} />,
    )
    const textarea = screen.getByPlaceholderText('Describe this workspace')
    fireEvent.change(textarea, { target: { value: 'New desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('New desc')
  })

  it('displays current name value', () => {
    render(
      <WorkspaceGeneralCard name="My Workspace" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByDisplayValue('My Workspace')).toBeDefined()
  })
})
