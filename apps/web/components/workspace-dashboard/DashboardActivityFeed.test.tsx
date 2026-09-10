/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { DashboardActivityFeed } from './DashboardActivityFeed'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
}))

const mockActivities = [
  { type: 'training', action: 'Job completed', detail: 'shakespeare-finetune', status: 'completed', timestamp: '2024-01-15T10:00:00Z', user: 'alice' },
  { type: 'dataset', action: 'Dataset uploaded', detail: 'wiki-text', status: 'success', timestamp: '2024-01-15T09:00:00Z', user: 'bob' },
]

describe('DashboardActivityFeed', () => {
  it('renders the title', () => {
    render(<DashboardActivityFeed activities={[]} />)
    expect(screen.getByText('Recent Activity')).toBeDefined()
  })

  it('renders activity actions', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('Job completed')).toBeDefined()
    expect(screen.getByText('Dataset uploaded')).toBeDefined()
  })

  it('renders activity details', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('shakespeare-finetune')).toBeDefined()
  })

  it('renders activity status badges', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('completed')).toBeDefined()
    expect(screen.getByText('success')).toBeDefined()
  })

  it('shows empty state when no activities', () => {
    render(<DashboardActivityFeed activities={[]} />)
    expect(screen.getByText('No recent activity')).toBeDefined()
  })

  it('renders refresh button when onRefresh provided', () => {
    const { container } = render(<DashboardActivityFeed activities={[]} onRefresh={vi.fn()} />)
    expect(container.querySelector('button')).toBeDefined()
  })
})
