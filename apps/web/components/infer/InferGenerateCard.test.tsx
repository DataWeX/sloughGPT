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
  Input: (props: any) => <input {...props} />,
  Textarea: (props: any) => <textarea {...props} />,
  Label: ({ children, ...props }: any) => <label {...props}>{children}</label>,
}))

import { InferGenerateCard } from './InferGenerateCard'

afterEach(() => cleanup())

const defaultProps = {
  prompt: '',
  maxTokens: 128,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  repPenalty: 1.1,
  result: null,
  loading: false,
  onPromptChange: vi.fn(),
  onMaxTokensChange: vi.fn(),
  onTemperatureChange: vi.fn(),
  onTopPChange: vi.fn(),
  onTopKChange: vi.fn(),
  onRepPenaltyChange: vi.fn(),
  onRun: vi.fn(),
}

describe('InferGenerateCard', () => {
  it('renders title', () => {
    render(<InferGenerateCard {...defaultProps} />)
    expect(screen.getByText('Text Generation')).toBeTruthy()
  })

  it('renders prompt textarea', () => {
    render(<InferGenerateCard {...defaultProps} />)
    expect(screen.getByLabelText('Prompt')).toBeTruthy()
  })

  it('renders all parameter inputs', () => {
    render(<InferGenerateCard {...defaultProps} />)
    expect(screen.getByLabelText('Max Tokens')).toBeTruthy()
    expect(screen.getByLabelText('Temperature')).toBeTruthy()
    expect(screen.getByLabelText('Top P')).toBeTruthy()
    expect(screen.getByLabelText('Top K')).toBeTruthy()
    expect(screen.getByLabelText('Rep Penalty')).toBeTruthy()
  })

  it('renders run button', () => {
    render(<InferGenerateCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
  })

  it('shows result text when provided', () => {
    const result = {
      text: 'Hello world',
      tokens_generated: 5,
      elapsed_ms: 120,
      model: 'test-model',
    }
    render(<InferGenerateCard {...defaultProps} result={result} />)
    expect(screen.getByText('Hello world')).toBeTruthy()
  })

  it('shows result stats when provided', () => {
    const result = {
      text: 'output',
      tokens_generated: 10,
      elapsed_ms: 200,
      model: 'gpt-2',
    }
    render(<InferGenerateCard {...defaultProps} result={result} />)
    expect(screen.getByText('10 tokens')).toBeTruthy()
    expect(screen.getByText('200ms')).toBeTruthy()
    expect(screen.getByText('gpt-2')).toBeTruthy()
  })

  it('disables run button when loading', () => {
    render(<InferGenerateCard {...defaultProps} loading={true} />)
    const button = screen.getByRole('button', { name: /Generating/ })
    expect(button.hasAttribute('disabled')).toBe(true)
  })
})
