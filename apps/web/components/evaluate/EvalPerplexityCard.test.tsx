// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Textarea: (props: any) => <textarea {...props} />,
}))

import { EvalPerplexityCard } from './EvalPerplexityCard'

afterEach(() => cleanup())

describe('EvalPerplexityCard', () => {
  it('renders title', () => {
    render(<EvalPerplexityCard />)
    expect(screen.getByText('Perplexity Calculator')).toBeTruthy()
  })

  it('renders textarea', () => {
    render(<EvalPerplexityCard />)
    expect(screen.getByPlaceholderText('Enter text to calculate perplexity...')).toBeTruthy()
  })

  it('renders calculate button', () => {
    render(<EvalPerplexityCard />)
    expect(screen.getByRole('button', { name: /Calculate/ })).toBeTruthy()
  })

  it('disables calculate when empty', () => {
    render(<EvalPerplexityCard />)
    expect(screen.getByRole('button', { name: /Calculate/ }).getAttribute('disabled')).not.toBeNull()
  })

  it('shows results when provided', () => {
    render(<EvalPerplexityCard result={{ perplexity: 42.5, loss: 3.75, tokens: 128 }} />)
    expect(screen.getByText('42.5')).toBeTruthy()
    expect(screen.getByText('3.75')).toBeTruthy()
    expect(screen.getByText('128')).toBeTruthy()
  })
})
