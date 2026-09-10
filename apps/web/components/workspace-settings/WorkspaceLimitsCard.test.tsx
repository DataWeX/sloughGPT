/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceLimitsCard } from './WorkspaceLimitsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="limits-input" {...props} />,
}))

describe('WorkspaceLimitsCard', () => {
  const defaultProps = {
    defaultModel: 'llama-3.2-3b',
    dataRetentionDays: 90,
    maxMembers: 50,
    allowSharing: true,
    onDefaultModelChange: vi.fn(),
    onDataRetentionChange: vi.fn(),
    onMaxMembersChange: vi.fn(),
    onAllowSharingChange: vi.fn(),
  }

  it('renders the Defaults title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Defaults')).toBeDefined()
  })

  it('renders the Limits title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Limits')).toBeDefined()
  })

  it('renders the Sharing title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Sharing')).toBeDefined()
  })

  it('renders default model input', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByDisplayValue('llama-3.2-3b')).toBeDefined()
  })

  it('renders data retention input', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    const inputs = screen.getAllByTestId('limits-input')
    expect(inputs.some((el) => (el as HTMLInputElement).value === '90')).toBe(true)
  })

  it('renders allow sharing checkbox', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Allow data sharing')).toBeDefined()
  })

  it('calls onAllowSharingChange when checkbox toggled', () => {
    const onAllowSharingChange = vi.fn()
    render(<WorkspaceLimitsCard {...defaultProps} onAllowSharingChange={onAllowSharingChange} />)
    fireEvent.click(screen.getByText('Allow data sharing').closest('label')!)
    expect(onAllowSharingChange).toHaveBeenCalledOnce()
  })

  it('renders Save button when onSave provided', () => {
    render(<WorkspaceLimitsCard {...defaultProps} onSave={vi.fn()} />)
    expect(screen.getByText('Save Settings')).toBeDefined()
  })

  it('shows unsaved changes when hasChanges is true', () => {
    render(<WorkspaceLimitsCard {...defaultProps} onSave={vi.fn()} hasChanges />)
    expect(screen.getByText('You have unsaved changes')).toBeDefined()
  })
})
