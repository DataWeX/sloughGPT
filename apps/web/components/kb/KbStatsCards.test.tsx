import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { KbStatsCards } from './KbStatsCards'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

describe('KbStatsCards', () => {
  it('renders nothing when stats is null', () => {
    const { container } = render(<KbStatsCards stats={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders all four stat labels', () => {
    render(<KbStatsCards stats={{ total_items: 42, topics: ['a', 'b'], avg_importance: 0.8, source_count: 3 }} />)
    expect(screen.getByText('Total Entries')).toBeDefined()
    expect(screen.getByText('Topics')).toBeDefined()
    expect(screen.getByText('Avg Importance')).toBeDefined()
    expect(screen.getByText('Sources')).toBeDefined()
  })

  it('displays the correct values', () => {
    render(<KbStatsCards stats={{ total_items: 10, topics: ['x'], avg_importance: 0.5, source_count: 2 }} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('0.50')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('renders four card components', () => {
    render(<KbStatsCards stats={{ total_items: 0, topics: [], avg_importance: 0, source_count: 0 }} />)
    expect(screen.getAllByTestId('card')).toHaveLength(4)
  })

  it('formats importance to two decimal places', () => {
    render(<KbStatsCards stats={{ total_items: 1, topics: [], avg_importance: 0.123, source_count: 0 }} />)
    expect(screen.getByText('0.12')).toBeDefined()
  })
})
