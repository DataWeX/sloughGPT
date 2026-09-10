/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceGeneralSettingsCard } from './WorkspaceGeneralSettingsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="settings-input" {...props} />,
}))

describe('WorkspaceGeneralSettingsCard', () => {
  const defaultProps = {
    name: 'My Workspace',
    description: 'A test workspace',
    onNameChange: vi.fn(),
    onDescriptionChange: vi.fn(),
  }

  it('renders the title', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect(screen.getByText('General')).toBeDefined()
  })

  it('renders name input with value', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect((screen.getByTestId('settings-input') as HTMLInputElement).value).toBe('My Workspace')
  })

  it('renders description textarea', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect(screen.getByDisplayValue('A test workspace')).toBeDefined()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onNameChange={onNameChange} />)
    fireEvent.change(screen.getByTestId('settings-input'), { target: { value: 'New Name' } })
    expect(onNameChange).toHaveBeenCalledWith('New Name')
  })

  it('calls onDescriptionChange when description changes', () => {
    const onDescriptionChange = vi.fn()
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onDescriptionChange={onDescriptionChange} />)
    fireEvent.change(screen.getByDisplayValue('A test workspace'), { target: { value: 'New desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('New desc')
  })

  it('renders Save button when onSave provided', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} />)
    expect(screen.getByText('Save')).toBeDefined()
  })

  it('shows unsaved changes text when hasChanges is true', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} hasChanges />)
    expect(screen.getByText('Unsaved changes')).toBeDefined()
  })

  it('disables Save when no changes', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} hasChanges={false} />)
    expect(screen.getByText('Save')).toHaveProperty('disabled', true)
  })
})
