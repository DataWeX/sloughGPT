import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const mockList = vi.fn()
const mockDelete = vi.fn()
const mockPreview = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: (...args: unknown[]) => mockList(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    preview: (...args: unknown[]) => mockPreview(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import DatasetsPage from './page'

describe('DatasetsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
  })

  afterEach(() => {
    cleanup()
  })

  it('renders page header', async () => {
    render(<DatasetsPage />)
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no datasets', async () => {
    render(<DatasetsPage />)
    await screen.findAllByText(/no datasets yet/i)
  })

  it('displays datasets from controller', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'shakespeare', format: 'text', rows: 5, size_bytes: 1024, created_at: '2026-01-01' },
    ])
    render(<DatasetsPage />)
    await screen.findAllByText('shakespeare')
  })
})
