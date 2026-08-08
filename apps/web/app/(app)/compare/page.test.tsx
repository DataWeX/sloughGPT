import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: vi.fn().mockResolvedValue([]),
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', model_type: null }),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    runBenchmark: vi.fn(),
    getHistory: vi.fn().mockResolvedValue([]),
  },
}))

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mockAddToast }) => unknown) => selector({ addToast: mockAddToast }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
  getJsonItem: vi.fn().mockReturnValue([]),
}))

vi.mock('next/dynamic', () => {
  const React = require('react')
  return {
    __esModule: true,
    default: () => (props: Record<string, unknown>) => React.createElement('div', { 'data-testid': 'dynamic' }),
  }
})

import ComparePage from './page'

describe('ComparePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders page header', () => {
    render(<ComparePage />)
    expect(screen.getAllByText('Model Comparison').length).toBeGreaterThanOrEqual(1)
  })
})
