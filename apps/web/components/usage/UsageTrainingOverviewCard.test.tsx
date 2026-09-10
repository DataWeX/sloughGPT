/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { UsageTrainingOverviewCard, type TrainingStatus } from './UsageTrainingOverviewCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const status: TrainingStatus = {
  completed: 10,
  running: 2,
  queued: 5,
  failed: 1,
}

describe('UsageTrainingOverviewCard', () => {
  it('renders the card title', () => {
    render(<UsageTrainingOverviewCard status={status} totalMinutes={0} />)
    expect(screen.getByText('Training Jobs')).toBeDefined()
  })

  it('renders completed count', () => {
    render(<UsageTrainingOverviewCard status={status} totalMinutes={0} />)
    expect(screen.getByText('10')).toBeDefined()
  })

  it('renders all status labels', () => {
    render(<UsageTrainingOverviewCard status={status} totalMinutes={0} />)
    expect(screen.getByText('Completed')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
    expect(screen.getByText('Queued')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })

  it('renders total minutes', () => {
    render(<UsageTrainingOverviewCard status={status} totalMinutes={42} />)
    expect(screen.getByText('Total training time: 42 minutes')).toBeDefined()
  })

  it('renders zero counts correctly', () => {
    render(<UsageTrainingOverviewCard status={{ completed: 0, running: 0, queued: 0, failed: 0 }} totalMinutes={0} />)
    expect(screen.getByText('Total training time: 0 minutes')).toBeDefined()
  })
})
