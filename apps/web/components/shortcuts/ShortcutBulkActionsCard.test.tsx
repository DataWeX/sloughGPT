/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutBulkActionsCard, type BulkAction } from './ShortcutBulkActionsCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
}))

afterEach(() => cleanup())

const actions: BulkAction[] = [
  { label: 'Refresh All', onClick: vi.fn() },
  { label: 'Export', onClick: vi.fn() },
]

describe('ShortcutBulkActionsCard', () => {
  it('renders the title', () => {
    render(<ShortcutBulkActionsCard actions={[]} />)
    expect(screen.getByText('Quick Actions')).toBeDefined()
  })

  it('renders action buttons', () => {
    render(<ShortcutBulkActionsCard actions={actions} />)
    expect(screen.getByText('Refresh All')).toBeDefined()
    expect(screen.getByText('Export')).toBeDefined()
  })

  it('calls onClick when action button is clicked', () => {
    render(<ShortcutBulkActionsCard actions={actions} />)
    fireEvent.click(screen.getByText('Refresh All'))
    expect(actions[0].onClick).toHaveBeenCalledOnce()
  })

  it('renders empty actions gracefully', () => {
    render(<ShortcutBulkActionsCard actions={[]} />)
    expect(screen.getByText('Quick Actions')).toBeDefined()
    const content = screen.getByTestId('card-content')
    expect(content.querySelectorAll('[data-testid="button"]').length).toBe(0)
  })

  it('disables buttons when disabled prop is true', () => {
    render(<ShortcutBulkActionsCard actions={actions} disabled />)
    const buttons = screen.getAllByTestId('button')
    buttons.forEach(btn => expect(btn).toBeDisabled())
  })
})
