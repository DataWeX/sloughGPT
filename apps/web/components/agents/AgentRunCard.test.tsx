// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}))

import { AgentRunCard } from './AgentRunCard'

afterEach(() => cleanup())

const mockRuns = [
  {
    id: 'run-1',
    status: 'completed' as const,
    task: 'Summarize the document',
    result: 'The document covers AI safety...',
    started_at: '2026-09-10T10:00:00Z',
    completed_at: '2026-09-10T10:02:00Z',
  },
  {
    id: 'run-2',
    status: 'failed' as const,
    task: 'Translate text to French',
    started_at: '2026-09-10T09:50:00Z',
    completed_at: '2026-09-10T09:50:05Z',
  },
  {
    id: 'run-3',
    status: 'running' as const,
    task: 'Generate code review',
    started_at: '2026-09-10T10:05:00Z',
  },
]

describe('AgentRunCard', () => {
  it('renders title', () => {
    render(<AgentRunCard runs={[]} />)
    expect(screen.getByText('Recent Runs')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentRunCard runs={[]} />)
    expect(screen.getByText('No runs yet.')).toBeTruthy()
  })

  it('renders run tasks', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getByText('Summarize the document')).toBeTruthy()
    expect(screen.getByText('Translate text to French')).toBeTruthy()
    expect(screen.getByText('Generate code review')).toBeTruthy()
  })

  it('renders run statuses', () => {
    render(<AgentRunCard runs={mockRuns} />)
    const badges = screen.getAllByText('completed')
    expect(badges.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('failed')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
  })

  it('renders result preview', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getByText('The document covers AI safety...')).toBeTruthy()
  })

  it('renders timestamps', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getAllByText(/Started/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Completed/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders status dots', () => {
    const { container } = render(<AgentRunCard runs={mockRuns} />)
    const dots = container.querySelectorAll('.rounded-full')
    expect(dots.length).toBe(3)
  })
})
