import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <h3>{children}</h3>,
    Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
  }
})

vi.mock('lucide-react', () => ({
  GitCompare: () => <span />,
  ArrowRight: () => <span />,
  Trophy: () => <span />,
}))

const { mockComparePair } = vi.hoisted(() => ({ mockComparePair: vi.fn() }))

vi.mock('@/hooks/useRunCompare', () => ({
  compareMetric: (field: string, a: number, b: number) => (a === b ? 'tie' : 'a'),
  useRunCompare: () => ({
    runOptions: [
      { run_id: 'a', model: 'gpt2', quality_score: 0.9 },
      { run_id: 'b', model: 'llama', quality_score: 0.8 },
    ],
    fetchingOptions: false,
    runA: 'a',
    runB: 'b',
    setRunA: vi.fn(),
    setRunB: vi.fn(),
    result: null,
    comparing: false,
    compare: vi.fn(),
    comparePair: mockComparePair,
  }),
}))

import { CompareSection } from './CompareSection'

afterEach(cleanup)
beforeEach(() => { vi.clearAllMocks() })

describe('CompareSection', () => {
  it('renders selectors and calls comparePair on Compare', async () => {
    render(<CompareSection />)
    expect(screen.getByText('Select two training runs to compare them side by side.')).toBeTruthy()
    await act(async () => { fireEvent.click(screen.getByText('Compare')) })
    expect(mockComparePair).toHaveBeenCalledWith('a', 'b')
  })
})
