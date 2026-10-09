/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect } from 'vitest'
import { EvalHistoryCard } from './EvalHistoryCard'

afterEach(() => cleanup())

describe('EvalHistoryCard', () => {
  it('renders title with count', () => {
    render(<EvalHistoryCard history={[]} loading={false} />)
    expect(screen.getByText('Eval History (0)')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<EvalHistoryCard history={[]} loading={true} />)
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('shows empty state', () => {
    render(<EvalHistoryCard history={[]} loading={false} />)
    expect(screen.getByText('No evaluations yet. Run an eval above.')).toBeInTheDocument()
  })

  it('displays eval status when data provided', () => {
    const history = [
      { status: 'passed', elapsed_ms: 1200, report: 'All tests passed' },
    ]
    render(<EvalHistoryCard history={history} loading={false} />)
    expect(screen.getByText('passed')).toBeInTheDocument()
  })

  it('displays elapsed time', () => {
    const history = [
      { status: 'passed', elapsed_ms: 1200, report: 'All tests passed' },
    ]
    render(<EvalHistoryCard history={history} loading={false} />)
    expect(screen.getByText('1200ms')).toBeInTheDocument()
  })

  it('displays report text', () => {
    const history = [
      { status: 'passed', elapsed_ms: 1200, report: 'All tests passed' },
    ]
    render(<EvalHistoryCard history={history} loading={false} />)
    expect(screen.getByText('All tests passed')).toBeInTheDocument()
  })
})
