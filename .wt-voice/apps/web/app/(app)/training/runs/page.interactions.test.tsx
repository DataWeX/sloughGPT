/**
 * Interaction tests for the training runs page: the seams created by the
 * Phase 2 split (page → useTrainingRuns + RunList + RunDetailsPanel) —
 * card selection, notes editing, bulk selection, compare retargeting,
 * and tag entry.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, className }: any) => (
      <button onClick={onClick} disabled={disabled} className={className}>
        {children}
      </button>
    ),
    Card: ({ children, className, onClick }: any) => (
      <div className={className} onClick={onClick}>
        {children}
      </div>
    ),
    CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
    Badge: ({ children, className }: any) => <span className={className}>{children}</span>,
    StatusBadge: ({ children }: any) => <span>{children}</span>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => (
      <div>
        <span>{label}</span>
        <span>{String(value)}</span>
      </div>
    ),
    SectionHeader: ({ title, description }: any) => (
      <div>
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
    ),
    Input: ({ value, onChange, placeholder, className, onKeyDown, 'aria-label': ariaLabel }: any) => (
      <input
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={className}
        onKeyDown={onKeyDown}
        aria-label={ariaLabel}
      />
    ),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, headerRight, toolbar }: any) => (
    <div data-testid="page-container" data-title={title}>
      <h1>{title}</h1>
      {toolbar}
      {headerRight}
      {children}
    </div>
  ),
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
const mockSetRunNotes = vi.fn()
const mockAddRunTag = vi.fn()
const mockCompareTrainingRuns = vi.fn()

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    filterTrainingRuns: (...args: any[]) => mockFilterTrainingRuns(...args),
    exportTrainingHistory: vi.fn(() => Promise.resolve({ content: '' })),
    deleteTrainingRun: vi.fn(() => Promise.resolve({})),
    addRunTag: (...args: any[]) => mockAddRunTag(...args),
    removeRunTag: vi.fn(() => Promise.resolve({})),
    setRunNotes: (...args: any[]) => mockSetRunNotes(...args),
    exportTrainingRun: vi.fn(() => Promise.resolve({ content: '' })),
    compareTrainingRuns: (...args: any[]) => mockCompareTrainingRuns(...args),
    toggleBookmark: vi.fn(() => Promise.resolve({})),
    duplicateTrainingRun: vi.fn(() => Promise.resolve({})),
    bulkDeleteRuns: vi.fn(() => Promise.resolve({})),
    bulkAddTag: vi.fn(() => Promise.resolve({})),
    bulkBookmark: vi.fn(() => Promise.resolve({})),
  },
}))

import TrainingRunsPage from './page'

const RUN_A = {
  run_id: 'run-1',
  model: 'gpt2',
  method: 'lora',
  converged: true,
  quality_score: 0.85,
  timestamp: 1700000000,
  training_time_s: 120,
  epochs: 3,
  final_loss: 0.5,
  tags: ['prod'],
  notes: 'baseline notes',
}
const RUN_B = {
  run_id: 'run-2',
  model: 'llama',
  method: 'qlora',
  timestamp: 1700000100,
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

beforeEach(() => {
  vi.clearAllMocks()
  mockFilterTrainingRuns.mockResolvedValue({ runs: [RUN_A, RUN_B] })
  mockSetRunNotes.mockResolvedValue({})
  mockAddRunTag.mockResolvedValue({})
  mockCompareTrainingRuns.mockResolvedValue({
    differences: { final_loss: { run_a: 0.5, run_b: 0.6 } },
  })
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:fake'), revokeObjectURL: vi.fn() })
  vi.stubGlobal(
    'confirm',
    vi.fn(() => true),
  )
})

async function renderLoaded() {
  render(<TrainingRunsPage />)
  await waitFor(() => {
    expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(1)
  })
}

async function openPanel() {
  await renderLoaded()
  // Click the card's unique "converged" badge — "gpt2" also appears in the
  // model filter's <option> list, so it cannot address the card alone.
  fireEvent.click(screen.getByText('converged'))
  await waitFor(() => {
    expect(screen.getByText('Run Details')).toBeTruthy()
  })
}

describe('TrainingRunsPage interactions', () => {
  it('opens the details panel when a run card is clicked', async () => {
    await openPanel()
    expect(screen.getByText('run-1')).toBeTruthy()
    expect(screen.getByText('baseline notes')).toBeTruthy()
  })

  it('edits and saves run notes through the panel', async () => {
    await openPanel()
    fireEvent.click(screen.getByText('Edit'))
    const textarea = screen.getByPlaceholderText('Add notes about this training run...')
    expect((textarea as HTMLTextAreaElement).value).toBe('baseline notes')
    fireEvent.change(textarea, { target: { value: 'updated notes' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => {
      expect(mockSetRunNotes).toHaveBeenCalledWith('run-1', 'updated notes')
    })
  })

  it('selects all runs, shows the bulk bar, and clears the selection', async () => {
    await renderLoaded()
    const selectAllButton = screen.getByText('Select all').previousElementSibling as HTMLElement
    fireEvent.click(selectAllButton)
    expect(screen.getByText('2 selected')).toBeTruthy()
    fireEvent.click(screen.getByText('Clear Selection'))
    await waitFor(() => {
      expect(screen.queryByText('2 selected')).toBeNull()
    })
  })

  it('runs a compare and clears the result when the target changes', async () => {
    await openPanel()
    fireEvent.change(screen.getByDisplayValue('Select run to compare...'), {
      target: { value: 'run-2' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))
    await waitFor(() => {
      expect(screen.getByText('Differences:')).toBeTruthy()
      expect(mockCompareTrainingRuns).toHaveBeenCalledWith('run-1', 'run-2')
    })
    fireEvent.change(screen.getByDisplayValue('llama - run-2'), { target: { value: '' } })
    await waitFor(() => {
      expect(screen.queryByText('Differences:')).toBeNull()
    })
  })

  it('adds a tag to the selected run via Enter', async () => {
    await openPanel()
    const tagInput = screen.getByPlaceholderText('Add tag...')
    fireEvent.change(tagInput, { target: { value: 'v2' } })
    fireEvent.keyDown(tagInput, { key: 'Enter' })
    await waitFor(() => {
      expect(mockAddRunTag).toHaveBeenCalledWith('run-1', 'v2')
    })
  })
})
