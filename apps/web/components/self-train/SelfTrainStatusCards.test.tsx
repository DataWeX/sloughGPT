// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

import { SelfTrainStatusCards } from './SelfTrainStatusCards'

describe('SelfTrainStatusCards', () => {
  afterEach(() => cleanup())

  it('renders title labels', () => {
    render(<SelfTrainStatusCards />)
    expect(screen.getByText('Status')).toBeTruthy()
    expect(screen.getByText('PID')).toBeTruthy()
    expect(screen.getByText('Exit code')).toBeTruthy()
    expect(screen.getByText('History lines')).toBeTruthy()
  })

  it('shows "Not started" when no status is provided', () => {
    render(<SelfTrainStatusCards />)
    expect(screen.getByText('Not started')).toBeTruthy()
  })

  it('shows "Running" when status is running', () => {
    render(<SelfTrainStatusCards status="running" />)
    expect(screen.getByText('Running')).toBeTruthy()
  })

  it('shows "Exited" when status is exited', () => {
    render(<SelfTrainStatusCards status="exited" />)
    expect(screen.getByText('Exited')).toBeTruthy()
  })

  it('shows loading state when loading is true', () => {
    render(<SelfTrainStatusCards loading />)
    const cards = screen.getAllByTestId('card')
    expect(within(cards[0]).getByText('...')).toBeTruthy()
  })

  it('renders pid and returncode when provided', () => {
    render(<SelfTrainStatusCards pid={12345} returncode={42} historyCount={5} />)
    expect(screen.getByText('12345')).toBeTruthy()
    expect(screen.getByText('42')).toBeTruthy()
  })

  it('renders history count', () => {
    render(<SelfTrainStatusCards historyCount={42} />)
    expect(screen.getByText('42')).toBeTruthy()
  })

  it('renders four cards', () => {
    const { container } = render(<SelfTrainStatusCards />)
    const cards = container.querySelectorAll('[data-testid="card"]')
    expect(cards).toHaveLength(4)
  })
})
