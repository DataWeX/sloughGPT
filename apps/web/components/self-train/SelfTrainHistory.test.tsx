// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
}))

import { SelfTrainHistory } from './SelfTrainHistory'

describe('SelfTrainHistory', () => {
  afterEach(() => cleanup())

  it('renders title "Training history"', () => {
    render(<SelfTrainHistory />)
    expect(screen.getByText('Training history')).toBeTruthy()
  })

  it('shows "No history yet" when history is empty', () => {
    render(<SelfTrainHistory history={[]} />)
    expect(screen.getByText('No history yet.')).toBeTruthy()
  })

  it('shows "No history yet" when history is undefined', () => {
    render(<SelfTrainHistory />)
    expect(screen.getByText('No history yet.')).toBeTruthy()
  })

  it('renders history lines when provided', () => {
    render(<SelfTrainHistory history={['epoch 1 loss: 0.5', 'epoch 2 loss: 0.3']} />)
    expect(screen.getByText('epoch 1 loss: 0.5')).toBeTruthy()
    expect(screen.getByText('epoch 2 loss: 0.3')).toBeTruthy()
  })

  it('renders multiple history lines', () => {
    const lines = Array.from({ length: 10 }, (_, i) => `line ${i}`)
    render(<SelfTrainHistory history={lines} />)
    lines.forEach(line => {
      expect(screen.getByText(line)).toBeTruthy()
    })
  })
})
