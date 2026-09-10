/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { DashboardTrainingOverview } from './DashboardTrainingOverview'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
}))

describe('DashboardTrainingOverview', () => {
  const defaultProps = {
    training: { completed: 5, running: 2, failed: 1 },
    totalMinutes: 120,
    totalJobs: 8,
  }

  it('renders the title', () => {
    render(<DashboardTrainingOverview {...defaultProps} />)
    expect(screen.getByText('Training Overview')).toBeDefined()
  })

  it('renders completed count', () => {
    render(<DashboardTrainingOverview {...defaultProps} />)
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders running count', () => {
    render(<DashboardTrainingOverview {...defaultProps} />)
    expect(screen.getByText('2')).toBeDefined()
  })

  it('renders failed count', () => {
    render(<DashboardTrainingOverview {...defaultProps} />)
    expect(screen.getByText('1')).toBeDefined()
  })

  it('renders total training time', () => {
    render(<DashboardTrainingOverview {...defaultProps} />)
    expect(screen.getByText('Total training time: 120 min')).toBeDefined()
  })

  it('renders refresh button when onRefresh provided', () => {
    const { container } = render(<DashboardTrainingOverview {...defaultProps} onRefresh={vi.fn()} />)
    expect(container.querySelector('button')).toBeDefined()
  })
})
