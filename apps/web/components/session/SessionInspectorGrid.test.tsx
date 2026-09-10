// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

import { SessionInspectorGrid } from './SessionInspectorGrid'

describe('SessionInspectorGrid', () => {
  afterEach(() => cleanup())

  it('renders nothing when stats is undefined', () => {
    const { container } = render(<SessionInspectorGrid />)
    expect(container.innerHTML).toBe('')
  })

  it('renders stat labels when stats are provided', () => {
    render(<SessionInspectorGrid stats={{ messages: 10, knowledgeFacts: 5, feedback: 3, elapsedMs: 120 }} />)
    expect(screen.getByText('Messages')).toBeTruthy()
    expect(screen.getByText('Knowledge Facts')).toBeTruthy()
    expect(screen.getByText('Feedback')).toBeTruthy()
    expect(screen.getByText('Inspect Time')).toBeTruthy()
  })

  it('renders stat values', () => {
    render(<SessionInspectorGrid stats={{ messages: 10, elapsedMs: 120 }} />)
    expect(screen.getByText('10')).toBeTruthy()
    expect(screen.getByText('120ms')).toBeTruthy()
  })

  it('renders Workspace and Modes & Traits cards', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} />)
    expect(screen.getByText('Workspace')).toBeTruthy()
    expect(screen.getByText('Modes & Traits')).toBeTruthy()
  })

  it('renders working memory items', () => {
    render(<SessionInspectorGrid stats={{ workingMemory: ['item1', 'item2'] }} />)
    expect(screen.getByText('item1')).toBeTruthy()
    expect(screen.getByText('item2')).toBeTruthy()
  })

  it('renders modes', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} modes={{ tone: 'friendly', style: 'concise' }} />)
    expect(screen.getByText('friendly')).toBeTruthy()
    expect(screen.getByText('concise')).toBeTruthy()
  })

  it('renders traits as JSON', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} traits={{ creativity: 0.8 }} />)
    expect(screen.getByText(/"creativity": 0.8/)).toBeTruthy()
  })
})
