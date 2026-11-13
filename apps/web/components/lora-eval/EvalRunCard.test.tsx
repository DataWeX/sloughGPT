/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { EvalRunCard } from './EvalRunCard'

afterEach(() => cleanup())

const defaultProps = {
  adapterPath: 'data/user_adapters/best.npz',
  soul: 'assistant',
  running: false,
  onAdapterPathChange: vi.fn(),
  onSoulChange: vi.fn(),
  onRun: vi.fn(),
}

describe('EvalRunCard', () => {
  it('renders title', () => {
    render(<EvalRunCard {...defaultProps} />)
    expect(screen.getByText('Run Evaluation')).toBeInTheDocument()
  })

  it('renders adapter path input', () => {
    render(<EvalRunCard {...defaultProps} />)
    expect(screen.getByDisplayValue('data/user_adapters/best.npz')).toBeInTheDocument()
  })

  it('renders soul input', () => {
    render(<EvalRunCard {...defaultProps} />)
    expect(screen.getByDisplayValue('assistant')).toBeInTheDocument()
  })

  it('renders Run Eval button', () => {
    render(<EvalRunCard {...defaultProps} />)
    expect(screen.getByText('Run Eval')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<EvalRunCard {...defaultProps} running={true} />)
    expect(screen.getByText('Running...')).toBeInTheDocument()
  })

  it('disables button when running', () => {
    render(<EvalRunCard {...defaultProps} running={true} />)
    expect(screen.getByText('Running...')).toBeDisabled()
  })
})
