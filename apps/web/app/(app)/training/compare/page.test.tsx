import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/training/compare',
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => <div data-testid="page-container" data-title={title}>{children}</div>,
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left }: any) => <div data-testid="app-route-header">{left}</div>,
  AppRouteHeaderLead: ({ title }: any) => <span>{title}</span>,
}))

vi.mock('lucide-react', () => ({
  GitCompare: () => <span data-testid="git-compare-icon" />,
  ArrowRight: () => <span data-testid="arrow-right-icon" />,
  Trophy: () => <span data-testid="trophy-icon" />,
  TrendingDown: () => <span data-testid="trending-down-icon" />,
  TrendingUp: () => <span data-testid="trending-up-icon" />,
}))

const mockExportTrainingHistory = vi.fn()
const mockCompareTrainingRuns = vi.fn()

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    exportTrainingHistory: (...args: any[]) => mockExportTrainingHistory(...args),
    compareTrainingRuns: (...args: any[]) => mockCompareTrainingRuns(...args),
  },
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Card: passthrough,
    CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardHeader: passthrough,
    CardTitle: ({ children }: any) => <h3>{children}</h3>,
    Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
    Input: ({ value, onChange, placeholder }: any) => <input value={value} onChange={onChange} placeholder={placeholder} />,
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
  }
})

import TrainingComparePage from './page'

describe('TrainingComparePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockExportTrainingHistory.mockResolvedValue({ outcomes: [] })
  })

  it('renders without crashing', async () => {
    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getByTestId('page-container')).toBeInTheDocument()
    })
  })

  it('renders page title', async () => {
    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getByTestId('page-container')).toHaveAttribute('data-title', 'Training Comparison')
    })
  })

  it('shows loading state while fetching runs', async () => {
    mockExportTrainingHistory.mockReturnValue(new Promise(() => {}))
    render(<TrainingComparePage />)
    const selects = screen.getAllByRole('combobox')
    expect(selects[0]).toBeDisabled()
    expect(selects[1]).toBeDisabled()
  })

  it('shows content after runs are loaded', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    render(<TrainingComparePage />)
    await waitFor(() => {
      const selects = screen.getAllByRole('combobox')
      expect(selects[0]).not.toBeDisabled()
    })
    expect(screen.getAllByText(/run-1/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/run-2/).length).toBeGreaterThan(0)
  })

  it('displays empty state when no comparison result', async () => {
    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getByText(/Select two training runs to compare them side by side/)).toBeInTheDocument()
    })
  })

  it('disables compare button when no runs selected', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Compare/ })).toBeDisabled()
    })
  })

  it('enables compare button after selecting two runs', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    mockCompareTrainingRuns.mockResolvedValue({
      run_a: { run_id: 'run-1', model: 'gpt2', dataset: 'ds1', method: 'sft', final_loss: 0.5, perplexity: 12.3, quality_score: 0.9, training_time_s: 300, converged: true },
      run_b: { run_id: 'run-2', model: 'gpt2-medium', dataset: 'ds1', method: 'sft', final_loss: 0.45, perplexity: 11.1, quality_score: 0.85, training_time_s: 600, converged: true },
      differences: { final_loss: { run_a: 0.5, run_b: 0.45 } },
      a_wins: 1,
      b_wins: 1,
    })

    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getAllByRole('combobox')[0]).not.toBeDisabled()
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'run-1' } })
    fireEvent.change(selects[1], { target: { value: 'run-2' } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Compare/ })).not.toBeDisabled()
    })
  })

  it('calls compareTrainingRuns when compare button is clicked', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    mockCompareTrainingRuns.mockResolvedValue({
      run_a: { run_id: 'run-1', model: 'gpt2', dataset: 'ds1', method: 'sft', final_loss: 0.5, perplexity: 12.3, quality_score: 0.9, training_time_s: 300, converged: true },
      run_b: { run_id: 'run-2', model: 'gpt2-medium', dataset: 'ds1', method: 'sft', final_loss: 0.45, perplexity: 11.1, quality_score: 0.85, training_time_s: 600, converged: true },
      differences: { final_loss: { run_a: 0.5, run_b: 0.45 } },
      a_wins: 1,
      b_wins: 1,
    })

    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getAllByRole('combobox')[0]).not.toBeDisabled()
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'run-1' } })
    fireEvent.change(selects[1], { target: { value: 'run-2' } })

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))

    await waitFor(() => {
      expect(mockCompareTrainingRuns).toHaveBeenCalledWith('run-1', 'run-2')
    })
  })

  it('shows comparison result after successful compare', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    mockCompareTrainingRuns.mockResolvedValue({
      run_a: { run_id: 'run-1', model: 'gpt2', dataset: 'ds1', method: 'sft', final_loss: 0.5, perplexity: 12.3, quality_score: 0.9, training_time_s: 300, converged: true },
      run_b: { run_id: 'run-2', model: 'gpt2-medium', dataset: 'ds1', method: 'sft', final_loss: 0.45, perplexity: 11.1, quality_score: 0.85, training_time_s: 600, converged: true },
      differences: { final_loss: { run_a: 0.5, run_b: 0.45 } },
      a_wins: 1,
      b_wins: 1,
    })

    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getAllByRole('combobox')[0]).not.toBeDisabled()
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'run-1' } })
    fireEvent.change(selects[1], { target: { value: 'run-2' } })

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))

    await waitFor(() => {
      expect(screen.getByText(/Tie/)).toBeInTheDocument()
      expect(screen.getAllByText(/Run A: run-1/).length).toBeGreaterThan(0)
      expect(screen.getAllByText(/Run B: run-2/).length).toBeGreaterThan(0)
      expect(screen.getAllByText(/gpt2/).length).toBeGreaterThan(0)
      expect(screen.getAllByText(/gpt2-medium/).length).toBeGreaterThan(0)
      expect(screen.getByText(/Differences/)).toBeInTheDocument()
    })
  })

  it('shows winner when run A wins', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    mockCompareTrainingRuns.mockResolvedValue({
      run_a: { run_id: 'run-1', model: 'gpt2', dataset: 'ds1', method: 'sft', final_loss: 0.5, perplexity: 12.3, quality_score: 0.9, training_time_s: 300, converged: true },
      run_b: { run_id: 'run-2', model: 'gpt2-medium', dataset: 'ds1', method: 'sft', final_loss: 0.45, perplexity: 11.1, quality_score: 0.85, training_time_s: 600, converged: false },
      differences: {},
      a_wins: 3,
      b_wins: 1,
    })

    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getAllByRole('combobox')[0]).not.toBeDisabled()
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'run-1' } })
    fireEvent.change(selects[1], { target: { value: 'run-2' } })

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))

    await waitFor(() => {
      expect(screen.getByText(/Run A wins \(3 vs 1\)/)).toBeInTheDocument()
    })
  })

  it('handles compare error gracefully', async () => {
    mockExportTrainingHistory.mockResolvedValue({
      outcomes: [
        { run_id: 'run-1', model: 'gpt2', quality_score: 0.9 },
        { run_id: 'run-2', model: 'gpt2-medium', quality_score: 0.85 },
      ],
    })
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockCompareTrainingRuns.mockRejectedValue(new Error('compare failed'))

    render(<TrainingComparePage />)
    await waitFor(() => {
      expect(screen.getAllByRole('combobox')[0]).not.toBeDisabled()
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'run-1' } })
    fireEvent.change(selects[1], { target: { value: 'run-2' } })

    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to compare:', expect.any(Error))
    })

    consoleSpy.mockRestore()
  })

  it('handles fetch runs error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    mockExportTrainingHistory.mockRejectedValue(new Error('fetch failed'))

    render(<TrainingComparePage />)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch runs:', expect.any(Error))
    })

    consoleSpy.mockRestore()
  })
})
