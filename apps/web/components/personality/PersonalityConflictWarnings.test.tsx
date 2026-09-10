/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityConflictWarnings } from './PersonalityConflictWarnings'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
}))

afterEach(() => cleanup())

const mockConflicts = [
  { type: 'voice', severity: 'medium', message: 'High formality conflicts with humor', fields: ['formality', 'humor'] },
  { type: 'trait', severity: 'low', message: 'Low openness may limit creativity', fields: ['openness'] },
]

describe('PersonalityConflictWarnings', () => {
  it('renders nothing when no conflicts', () => {
    const { container } = render(<PersonalityConflictWarnings conflicts={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders title when conflicts exist', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('Personality Conflicts')).toBeDefined()
  })

  it('renders conflict messages', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('High formality conflicts with humor')).toBeDefined()
    expect(screen.getByText('Low openness may limit creativity')).toBeDefined()
  })

  it('renders severity badges', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    const badges = screen.getAllByTestId('badge')
    expect(badges).toHaveLength(2)
    expect(screen.getByText('medium')).toBeDefined()
    expect(screen.getByText('low')).toBeDefined()
  })

  it('renders affected fields', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('formality + humor')).toBeDefined()
    expect(screen.getByText('openness')).toBeDefined()
  })

  it('renders description text', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('These settings may work against each other')).toBeDefined()
  })
})
