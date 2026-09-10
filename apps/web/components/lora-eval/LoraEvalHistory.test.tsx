import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { LoraEvalHistory } from './LoraEvalHistory'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

describe('LoraEvalHistory', () => {
  it('renders the history title with count', () => {
    render(<LoraEvalHistory history={[]} loading={false} />)
    expect(screen.getByText('Eval History (0)')).toBeDefined()
  })

  it('shows loading text when loading', () => {
    render(<LoraEvalHistory history={[]} loading={true} />)
    expect(screen.getByText('Loading...')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<LoraEvalHistory history={[]} loading={false} />)
    expect(screen.getByText('No evaluations yet. Run an eval above.')).toBeDefined()
  })

  it('displays history entries when provided', () => {
    const history = [
      { status: 'passed', elapsed_ms: 1200, report: 'All tests passed' },
      { status: 'failed', elapsed_ms: 800, report: 'Some tests failed' },
    ]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('passed')).toBeDefined()
    expect(screen.getByText('failed')).toBeDefined()
    expect(screen.getByText('1200ms')).toBeDefined()
    expect(screen.getByText('800ms')).toBeDefined()
  })

  it('renders report text when provided', () => {
    const history = [{ status: 'ok', report: 'Detailed report summary' }]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('Detailed report summary')).toBeDefined()
  })

  it('hides elapsed time when not provided', () => {
    const history = [{ status: 'done' }]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('done')).toBeDefined()
    expect(screen.queryByText(/ms/)).toBeNull()
  })
})
