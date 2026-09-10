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
}))

import { InferModelInfoCard } from './InferModelInfoCard'

afterEach(() => cleanup())

const defaultProps = {
  info: null,
  loading: false,
  onLoad: vi.fn(),
}

describe('InferModelInfoCard', () => {
  it('renders title', () => {
    render(<InferModelInfoCard {...defaultProps} />)
    expect(screen.getByText('Model Information')).toBeTruthy()
  })

  it('renders load button', () => {
    render(<InferModelInfoCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: /Load Info/ })).toBeTruthy()
  })

  it('disables button when loading', () => {
    render(<InferModelInfoCard {...defaultProps} loading={true} />)
    const button = screen.getByRole('button', { name: /Loading/ })
    expect(button.hasAttribute('disabled')).toBe(true)
  })

  it('shows stat boxes when info provided', () => {
    const info = {
      model_id: 'gpt-2',
      model_type: 'causal-lm',
      num_parameters: 124000000,
      vocab_size: 50257,
      max_context: 1024,
      num_layers: 12,
      has_tokenizer: true,
      has_streaming: true,
      has_embedding: false,
    }
    render(<InferModelInfoCard {...defaultProps} info={info} />)
    expect(screen.getByText('gpt-2')).toBeTruthy()
    expect(screen.getByText('causal-lm')).toBeTruthy()
    expect(screen.getByText('124,000,000')).toBeTruthy()
    expect(screen.getByText('50,257')).toBeTruthy()
    expect(screen.getByText('1,024')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
  })

  it('displays boolean stats as Yes/No', () => {
    const info = {
      model_id: 'm',
      model_type: 't',
      num_parameters: 1,
      vocab_size: 1,
      max_context: 1,
      num_layers: 1,
      has_tokenizer: true,
      has_streaming: false,
      has_embedding: true,
    }
    render(<InferModelInfoCard {...defaultProps} info={info} />)
    const yesNos = screen.getAllByText(/^(Yes|No)$/)
    expect(yesNos.length).toBe(3)
  })
})
