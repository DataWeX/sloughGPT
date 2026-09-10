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
  Textarea: (props: any) => <textarea {...props} />,
  Label: ({ children, ...props }: any) => <label {...props}>{children}</label>,
}))

import { InferEmbedCard } from './InferEmbedCard'

afterEach(() => cleanup())

const defaultProps = {
  prompt: '',
  result: null,
  loading: false,
  onPromptChange: vi.fn(),
  onRun: vi.fn(),
}

describe('InferEmbedCard', () => {
  it('renders title', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByText('Text Embedding')).toBeTruthy()
  })

  it('renders prompt textarea', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByLabelText('Embed prompt')).toBeTruthy()
  })

  it('renders run button', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
  })

  it('shows dimensions when result provided', () => {
    const result = {
      embedding: Array(384).fill(0.1),
      dimensions: 384,
      model: 'embed-model',
    }
    render(<InferEmbedCard {...defaultProps} result={result} />)
    expect(screen.getByText('384 dimensions')).toBeTruthy()
    expect(screen.getByText('embed-model')).toBeTruthy()
  })

  it('renders bar chart when result provided', () => {
    const result = {
      embedding: [0.5, -0.3, 0.8, 0.1, -0.6, 0.2, 0.4, -0.1, 0.7, 0.3, -0.4, 0.6, 0.2, -0.5, 0.9, 0.1, -0.2, 0.3, 0.5, -0.7, 0.4],
      dimensions: 384,
      model: 'embed-model',
    }
    render(<InferEmbedCard {...defaultProps} result={result} />)
    const chart = screen.getByLabelText('Embedding visualization')
    expect(chart.children.length).toBe(20)
  })

  it('disables run button when loading', () => {
    render(<InferEmbedCard {...defaultProps} loading={true} />)
    const button = screen.getByRole('button', { name: /Embedding/ })
    expect(button.hasAttribute('disabled')).toBe(true)
  })
})
