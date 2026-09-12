import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, className }: any) => <button onClick={onClick} disabled={disabled} className={className}>{children}</button>,
    Card: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
    Badge: ({ children, className }: any) => <span className={className}>{children}</span>,
    Input: ({ value, onChange, placeholder, className, onKeyDown }: any) => <input value={value} onChange={onChange} placeholder={placeholder} className={className} onKeyDown={onKeyDown} />,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => <div data-testid="page-container" data-title={title}>{children}</div>,
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => (
    <div data-testid="app-route-header"><div>{left}</div><div>{right}</div></div>
  ),
  AppRouteHeaderLead: ({ title }: any) => <span>{title}</span>,
}))

vi.mock('lucide-react', () => ({
  Clock: () => <span data-testid="icon-clock" />,
  Download: () => <span data-testid="icon-download" />,
  BarChart3: () => <span data-testid="icon-bar-chart" />,
  Trash2: () => <span data-testid="icon-trash" />,
  Search: () => <span data-testid="icon-search" />,
  X: () => <span data-testid="icon-x" />,
  Tag: () => <span data-testid="icon-tag" />,
  Plus: () => <span data-testid="icon-plus" />,
  GitCompare: () => <span data-testid="icon-compare" />,
  Star: () => <span data-testid="icon-star" />,
  Copy: () => <span data-testid="icon-copy" />,
  CheckSquare: () => <span data-testid="icon-check-square" />,
  Square: () => <span data-testid="icon-square" />,
}))

const mockFilterTrainingRuns = vi.fn()
const mockExportTrainingHistory = vi.fn()
const mockDeleteTrainingRun = vi.fn()
const mockAddRunTag = vi.fn()
const mockRemoveRunTag = vi.fn()
const mockSetRunNotes = vi.fn()
const mockExportTrainingRun = vi.fn()
const mockCompareTrainingRuns = vi.fn()
const mockToggleBookmark = vi.fn()
const mockDuplicateTrainingRun = vi.fn()
const mockBulkDeleteRuns = vi.fn()
const mockBulkAddTag = vi.fn()
const mockBulkBookmark = vi.fn()

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    filterTrainingRuns: (...args: any[]) => mockFilterTrainingRuns(...args),
    exportTrainingHistory: (...args: any[]) => mockExportTrainingHistory(...args),
    deleteTrainingRun: (...args: any[]) => mockDeleteTrainingRun(...args),
    addRunTag: (...args: any[]) => mockAddRunTag(...args),
    removeRunTag: (...args: any[]) => mockRemoveRunTag(...args),
    setRunNotes: (...args: any[]) => mockSetRunNotes(...args),
    exportTrainingRun: (...args: any[]) => mockExportTrainingRun(...args),
    compareTrainingRuns: (...args: any[]) => mockCompareTrainingRuns(...args),
    toggleBookmark: (...args: any[]) => mockToggleBookmark(...args),
    duplicateTrainingRun: (...args: any[]) => mockDuplicateTrainingRun(...args),
    bulkDeleteRuns: (...args: any[]) => mockBulkDeleteRuns(...args),
    bulkAddTag: (...args: any[]) => mockBulkAddTag(...args),
    bulkBookmark: (...args: any[]) => mockBulkBookmark(...args),
  },
}))

import TrainingRunsPage from './page'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

beforeEach(() => {
  vi.clearAllMocks()
  mockFilterTrainingRuns.mockResolvedValue({ runs: [] })
  mockExportTrainingHistory.mockResolvedValue({ content: '' })
  mockDeleteTrainingRun.mockResolvedValue({})
  mockAddRunTag.mockResolvedValue({})
  mockRemoveRunTag.mockResolvedValue({})
  mockSetRunNotes.mockResolvedValue({})
  mockExportTrainingRun.mockResolvedValue({ content: '' })
  mockCompareTrainingRuns.mockResolvedValue({ differences: {} })
  mockToggleBookmark.mockResolvedValue({})
  mockDuplicateTrainingRun.mockResolvedValue({})
  mockBulkDeleteRuns.mockResolvedValue({})
  mockBulkAddTag.mockResolvedValue({})
  mockBulkBookmark.mockResolvedValue({})
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:fake'), revokeObjectURL: vi.fn() })
  vi.stubGlobal('confirm', vi.fn(() => true))
  vi.stubGlobal('alert', vi.fn())
})

describe('TrainingRunsPage', () => {
  it('renders without crashing', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('page-container')).toBeTruthy()
    })
  })

  it('renders the page title', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('Training Runs')).toBeTruthy()
    })
  })

  it('shows loading state with skeleton cards', async () => {
    let resolvePromise: any
    mockFilterTrainingRuns.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TrainingRunsPage />)
    const cards = document.querySelectorAll('.animate-pulse')
    expect(cards.length).toBe(5)
    resolvePromise({ runs: [] })
  })

  it('shows "No training runs found" when data is empty', async () => {
    mockFilterTrainingRuns.mockResolvedValue({ runs: [] })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('No training runs found.')).toBeTruthy()
    })
  })

  it('renders training run cards after data loads', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [
        { run_id: 'run-1', model: 'gpt2', method: 'lora', converged: true, quality_score: 0.85, timestamp: 1700000000, training_time_s: 120, epochs: 3, final_loss: 0.5 },
        { run_id: 'run-2', model: 'llama', method: 'qlora', converged: false, timestamp: 1700000100, training_time_s: 45 },
      ],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('llama').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('lora').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('qlora').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('converged')).toBeTruthy()
  })

  it('displays summary stats correctly', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [
        { run_id: 'r1', converged: true, quality_score: 0.8, final_loss: 0.5 },
        { run_id: 'r2', converged: false, quality_score: 0.6, final_loss: 0.3 },
      ],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('Total Runs')).toBeTruthy()
      expect(screen.getByText('Converged')).toBeTruthy()
      expect(screen.getByText('Avg Quality')).toBeTruthy()
      expect(screen.getByText('Avg Loss')).toBeTruthy()
    })
  })

  it('calls filterTrainingRuns on mount', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(mockFilterTrainingRuns).toHaveBeenCalledWith({ limit: 200 })
    })
  })

  it('shows quality badge for runs with quality_score', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [{ run_id: 'r1', model: 'gpt2', quality_score: 0.9, timestamp: 1700000000 }],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('90%').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders export button', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('Export')).toBeTruthy()
    })
  })

  it('renders search input', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search runs, tags, notes...')).toBeTruthy()
    })
  })

  it('renders model and method filter dropdowns', async () => {
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('All Models')).toBeTruthy()
      expect(screen.getByText('All Methods')).toBeTruthy()
    })
  })

  it('handles error from filterTrainingRuns gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockFilterTrainingRuns.mockRejectedValue(new Error('fetch failed'))
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch training runs:', expect.any(Error))
    })
    expect(screen.getByText('No training runs found.')).toBeTruthy()
    consoleSpy.mockRestore()
  })

  it('shows tags on run cards when present', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [{ run_id: 'r1', model: 'gpt2', tags: ['production', 'v2'], timestamp: 1700000000 }],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('production')).toBeTruthy()
      expect(screen.getByText('v2')).toBeTruthy()
    })
  })

  it('shows notes on run cards when present', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [{ run_id: 'r1', model: 'gpt2', notes: 'good run', timestamp: 1700000000 }],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('"good run"')).toBeTruthy()
    })
  })

  it('shows training duration when available', async () => {
    mockFilterTrainingRuns.mockResolvedValue({
      runs: [{ run_id: 'r1', model: 'gpt2', training_time_s: 125, timestamp: 1700000000 }],
    })
    render(<TrainingRunsPage />)
    await waitFor(() => {
      expect(screen.getByText('2m 5s')).toBeTruthy()
    })
  })
})
