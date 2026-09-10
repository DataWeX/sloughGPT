/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { NotificationFeed } from './NotificationFeed'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const mockNotifications = [
  { type: 'training', title: 'Model trained', detail: 'GPT-4 finished fine-tuning', status: 'completed', timestamp: '2025-01-15T10:30:00Z' },
  { type: 'member', title: 'New member', detail: 'alice joined workspace', status: '', timestamp: '2025-01-15T11:00:00Z' },
]

describe('NotificationFeed', () => {
  it('renders title "Recent Events"', () => {
    render(<NotificationFeed notifications={[]} />)
    expect(screen.getByText('Recent Events')).toBeDefined()
  })

  it('shows empty message when no notifications', () => {
    render(<NotificationFeed notifications={[]} />)
    expect(screen.getByText('No notifications')).toBeDefined()
  })

  it('renders notification titles', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('Model trained')).toBeDefined()
    expect(screen.getByText('New member')).toBeDefined()
  })

  it('renders notification details', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('GPT-4 finished fine-tuning')).toBeDefined()
    expect(screen.getByText('alice joined workspace')).toBeDefined()
  })

  it('renders notification types', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('training')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('renders status badges when present', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('completed')).toBeDefined()
  })
})
