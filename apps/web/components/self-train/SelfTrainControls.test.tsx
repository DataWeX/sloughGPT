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
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

import { SelfTrainControls } from './SelfTrainControls'

const defaultProps = {
  model: '',
  temperature: 0.7,
  forever: false,
  isRunning: false,
  onModelChange: vi.fn(),
  onTemperatureChange: vi.fn(),
  onForeverToggle: vi.fn(),
  onStart: vi.fn(),
  onStop: vi.fn(),
}

describe('SelfTrainControls', () => {
  afterEach(() => cleanup())

  it('renders title "Controls"', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Controls')).toBeTruthy()
  })

  it('renders model and temperature inputs', () => {
    render(<SelfTrainControls {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs.length).toBe(2)
  })

  it('renders start button when not running', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Start self-training')).toBeTruthy()
  })

  it('renders stop button when running', () => {
    render(<SelfTrainControls {...defaultProps} isRunning />)
    expect(screen.getByText('Stop')).toBeTruthy()
  })

  it('shows "Starting..." when starting prop is true', () => {
    render(<SelfTrainControls {...defaultProps} starting />)
    expect(screen.getByText('Starting...')).toBeTruthy()
  })

  it('displays single pass label when forever is false', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Single pass')).toBeTruthy()
  })

  it('displays train forever label when forever is true', () => {
    render(<SelfTrainControls {...defaultProps} forever />)
    expect(screen.getByText('Train forever')).toBeTruthy()
  })
})
