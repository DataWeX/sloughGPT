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

import { SessionMessages } from './SessionMessages'

const messages = [
  { role: 'user', content: 'Hello' },
  { role: 'assistant', content: 'Hi there!' },
]

describe('SessionMessages', () => {
  afterEach(() => cleanup())

  it('renders title with message count', () => {
    render(<SessionMessages messages={messages} />)
    expect(screen.getByText('Recent Messages (2)')).toBeTruthy()
  })

  it('renders custom title when provided', () => {
    render(<SessionMessages messages={messages} title="Fetched Messages (2)" />)
    expect(screen.getByText('Fetched Messages (2)')).toBeTruthy()
  })

  it('renders message content', () => {
    render(<SessionMessages messages={messages} />)
    expect(screen.getByText('Hello')).toBeTruthy()
    expect(screen.getByText('Hi there!')).toBeTruthy()
  })

  it('renders role labels', () => {
    render(<SessionMessages messages={messages} />)
    expect(screen.getByText('user')).toBeTruthy()
    expect(screen.getByText('assistant')).toBeTruthy()
  })

  it('truncates content longer than 500 characters', () => {
    const longContent = 'x'.repeat(600)
    render(<SessionMessages messages={[{ role: 'user', content: longContent }]} />)
    expect(screen.getByText(longContent.slice(0, 500) + '...')).toBeTruthy()
  })

  it('renders empty list without errors', () => {
    render(<SessionMessages messages={[]} />)
    expect(screen.getByText('Recent Messages (0)')).toBeTruthy()
  })
})
