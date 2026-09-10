// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button data-testid="button" {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,
}))

import { SessionLoadCard } from './SessionLoadCard'

const defaultProps = {
  sessionId: '',
  onSessionIdChange: vi.fn(),
  onInspect: vi.fn(),
  onRegenerate: vi.fn(),
}

describe('SessionLoadCard', () => {
  afterEach(() => cleanup())

  it('renders title "Load Session"', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Load Session')).toBeTruthy()
  })

  it('renders session ID input', () => {
    render(<SessionLoadCard {...defaultProps} />)
    const input = screen.getByTestId('input')
    expect(input).toBeTruthy()
    expect(input).toHaveAttribute('placeholder', 'Enter session ID...')
  })

  it('renders Inspect button', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Inspect')).toBeTruthy()
  })

  it('renders Regenerate button', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Regenerate Last Response')).toBeTruthy()
  })

  it('disables Inspect when loading is true', () => {
    render(<SessionLoadCard {...defaultProps} loading />)
    const buttons = screen.getAllByTestId('button')
    expect(buttons[0]).toBeDisabled()
  })

  it('shows "Loading..." text when loading', () => {
    render(<SessionLoadCard {...defaultProps} loading />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows "Regenerating..." text when regenerating', () => {
    render(<SessionLoadCard {...defaultProps} regenerating />)
    expect(screen.getByText('Regenerating...')).toBeTruthy()
  })

  it('disables buttons when sessionId is empty', () => {
    render(<SessionLoadCard {...defaultProps} sessionId="" />)
    const buttons = screen.getAllByTestId('button')
    buttons.forEach(btn => expect(btn).toBeDisabled())
  })
})
