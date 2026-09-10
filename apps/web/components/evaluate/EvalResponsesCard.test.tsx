// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconRefresh: () => <span data-testid="icon-refresh" />,
}))

import { EvalResponsesCard } from './EvalResponsesCard'

afterEach(() => cleanup())

describe('EvalResponsesCard', () => {
  it('renders title with count', () => {
    render(<EvalResponsesCard responses={[]} />)
    expect(screen.getByText('Logged Responses (0)')).toBeTruthy()
  })

  it('shows empty state when no responses', () => {
    render(<EvalResponsesCard responses={[]} />)
    expect(screen.getByText('No responses logged yet.')).toBeTruthy()
  })

  it('renders responses', () => {
    const responses = [
      { user_message: 'Hello', assistant_response: 'Hi there', model: 'gpt2', tokens_generated: 10, duration_ms: 50, timestamp: '2024-01-01T00:00:00Z' },
      { user_message: 'How are you?', assistant_response: 'I am fine', model: 'gpt2', tokens_generated: 15, duration_ms: 75, timestamp: '2024-01-01T00:01:00Z' },
    ]
    render(<EvalResponsesCard responses={responses} />)
    expect(screen.getByText('Logged Responses (2)')).toBeTruthy()
    expect(screen.getByText('Hello')).toBeTruthy()
    expect(screen.getByText('Hi there')).toBeTruthy()
  })

  it('renders refresh and clear buttons when callbacks provided', () => {
    render(<EvalResponsesCard responses={[]} onRefresh={() => {}} onClear={() => {}} />)
    expect(screen.getByLabelText('Refresh responses')).toBeTruthy()
    expect(screen.getByText('Clear')).toBeTruthy()
  })
})
