import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TrainingOverviewCard } from './TrainingOverviewCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

const defaultProps = {
  completed: 10,
  running: 3,
  queued: 5,
  failed: 1,
  totalMinutes: 120,
}

describe('TrainingOverviewCard', () => {
  it('renders the card title', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('Training Jobs')).toBeDefined()
  })

  it('renders completed count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('Completed')).toBeDefined()
  })

  it('renders running count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
  })

  it('renders queued count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('Queued')).toBeDefined()
  })

  it('renders failed count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })

  it('renders total training minutes', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('Total training time: 120 minutes')).toBeDefined()
  })

  it('renders all stat values', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })
})
