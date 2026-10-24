import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Card: ({ children, onClick, className, ...props }: any) => (
      <div onClick={onClick} className={className} {...props}>{children}</div>
    ),
    CardContent: ({ children, className, ...props }: any) => (
      <div className={className} {...props}>{children}</div>
    ),
    Button: ({ children, onClick, disabled, className, ...props }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} {...props}>{children}</button>
    ),
    Badge: ({ children, className, ...props }: any) => <span className={className} {...props}>{children}</span>,
    Input: ({ value, onChange, placeholder, className, ...props }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} {...props} />
    ),
    Select: ({ children }: any) => <div>{children}</div>,
    SelectTrigger: ({ children }: any) => <div>{children}</div>,
    SelectValue: () => <span />,
    SelectContent: ({ children }: any) => <div>{children}</div>,
    SelectItem: ({ children }: any) => <span>{children}</span>,
    IconDownload: () => <span />,
    IconSearch: () => <span />,
  }
})

vi.mock('@/components/icons/NavIcons', () => ({
  IconClose: () => <span />,
}))

vi.mock('lucide-react', () => ({
  Clock: () => <span />,
  Download: () => <span />,
  BarChart3: () => <span />,
  Trash2: () => <span />,
  Search: () => <span />,
  X: () => <span />,
  Tag: () => <span />,
  Plus: () => <span />,
  Star: ({ className }: any) => <span className={className} />,
  Copy: () => <span />,
  CheckSquare: () => <span data-testid="checked" />,
  Square: () => <span data-testid="unchecked" />,
}))

const { mockRuns } = vi.hoisted(() => ({ mockRuns: vi.fn() }))

vi.mock('@/hooks/useTrainingRuns', () => ({
  useTrainingRuns: () => ({
    runs: [
      { run_id: 'r1', timestamp: 1767225600, model: 'gpt2', method: 'finetune', quality_score: 0.85, converged: true, training_time_s: 90, final_loss: 2.5, tags: ['good'], notes: 'nice run', bookmarked: false },
      { run_id: 'r2', timestamp: 1767225700, model: 'llama', method: 'lora', quality_score: 0.5, tags: [], bookmarked: true },
    ],
    loading: false,
    filterModel: '',
    filterMethod: '',
    setFilterModel: vi.fn(),
    setFilterMethod: vi.fn(),
    fetchRuns: vi.fn(),
    deleteRun: mockRuns.mockImplementation(async () => {}),
    duplicateRun: vi.fn(),
    toggleBookmark: vi.fn(),
    addTag: vi.fn().mockResolvedValue({}),
    removeTag: vi.fn().mockResolvedValue({}),
    saveNotes: vi.fn().mockResolvedValue({}),
    bulkDelete: vi.fn().mockResolvedValue(undefined),
    bulkTag: vi.fn().mockResolvedValue(undefined),
    bulkBookmark: vi.fn().mockResolvedValue(undefined),
  }),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: { exportTrainingHistory: vi.fn().mockResolvedValue({}) },
}))

import { RunsSection } from './RunsSection'

afterEach(cleanup)
beforeEach(() => { vi.clearAllMocks() })

describe('RunsSection', () => {
  it('renders summary and run rows', () => {
    render(<RunsSection />)
    expect(screen.getByText('Outcome history')).toBeTruthy()
    expect(screen.getAllByText('gpt2').length).toBeGreaterThan(0)
    expect(screen.getAllByText('llama').length).toBeGreaterThan(0)
    expect(screen.getByText('85%')).toBeTruthy()
  })

  it('filters runs by search query', async () => {
    render(<RunsSection />)
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Search runs'), { target: { value: 'llama' } })
    })
    expect(screen.queryByText('85%')).toBeNull()
    expect(screen.getAllByText('llama').length).toBeGreaterThan(0)
  })

  it('opens the detail panel with tags and notes', async () => {
    render(<RunsSection />)
    const rows = screen.getAllByText('gpt2')
    const row = rows.map(el => el.closest('div[class*="cursor-pointer"]')).find(Boolean)
    expect(row).toBeTruthy()
    await act(async () => { fireEvent.click(row!) })
    expect(screen.getByText('Run Details')).toBeTruthy()
    expect(screen.getByText('nice run')).toBeTruthy()
  })

  it('deletes a run after inline confirm', async () => {
    render(<RunsSection />)
    await act(async () => { fireEvent.click(screen.getAllByLabelText('Delete run')[0]) })
    await act(async () => { fireEvent.click(screen.getByText('Confirm')) })
    await waitFor(() => { expect(mockRuns).toHaveBeenCalledWith('r1') })
  })
})
