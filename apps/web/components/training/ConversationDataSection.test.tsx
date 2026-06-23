// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

const { mockTrainingJobs, mockAddToast } = vi.hoisted(() => ({
  mockTrainingJobs: {
    getStatus: vi.fn(),
    exportFeedbackPairs: vi.fn(),
  },
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: mockTrainingJobs,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/dev-log', () => ({
  devDebug: vi.fn(),
}))

import { ConversationDataSection } from './ConversationDataSection'

const defaultStats = { total_jobs: 50, completed_jobs: 20, running_jobs: [{}], failed_jobs: 5 }

describe('ConversationDataSection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockTrainingJobs.getStatus.mockResolvedValue(defaultStats)
  })

  afterEach(cleanup)

  it('shows loading state initially', () => {
    mockTrainingJobs.getStatus.mockReturnValue(new Promise(() => {}))
    render(<ConversationDataSection />)
    expect(screen.getByText('Loading...')).toBeDefined()
  })

  it('renders stat cards after fetch', async () => {
    render(<ConversationDataSection />)
    expect(await screen.findByText('20')).toBeDefined()
    expect(screen.getByText('Positive (👍)')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('Negative (👎)')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('Neutral')).toBeDefined()
    expect(screen.getByText('Total Pairs')).toBeDefined()
  })

  it('renders 3 strategy options', async () => {
    render(<ConversationDataSection />)
    await screen.findByText('20')
    const options = screen.getAllByRole('option')
    expect(options.length).toBe(3)
    expect(options[0].textContent).toContain('Balanced')
    expect(options[1].textContent).toContain('Weighted')
    expect(options[2].textContent).toContain('Simple')
  })

  it('renders target count input', async () => {
    render(<ConversationDataSection />)
    await screen.findByText('20')
    const input = screen.getByDisplayValue('100') as HTMLInputElement
    expect(input).toBeDefined()
  })

  it('renders export button disabled when < 5 pairs', async () => {
    mockTrainingJobs.getStatus.mockResolvedValue({ total_jobs: 2, completed_jobs: 0, running_jobs: [], failed_jobs: 0 })
    render(<ConversationDataSection />)
    const btn = await screen.findByText('Export 100 Pairs for Training') as HTMLButtonElement
    expect(btn.disabled).toBe(true)
  })

  it('shows warning when < 5 pairs', async () => {
    mockTrainingJobs.getStatus.mockResolvedValue({ total_jobs: 2, completed_jobs: 0, running_jobs: [], failed_jobs: 0 })
    render(<ConversationDataSection />)
    const warning = await screen.findByText(/Need at least 5 conversation pairs/)
    expect(warning).toBeDefined()
  })

  it('calls exportFeedbackPairs on export', async () => {
    mockTrainingJobs.exportFeedbackPairs.mockResolvedValue({ pairs_count: 100, filepath: '/tmp/export.jsonl' })
    render(<ConversationDataSection />)
    await screen.findByText('20')
    fireEvent.click(screen.getByText('Export 100 Pairs for Training'))
    expect(mockTrainingJobs.exportFeedbackPairs).toHaveBeenCalledWith(0, 100)
  })

  it('shows success toast on export', async () => {
    mockTrainingJobs.exportFeedbackPairs.mockResolvedValue({ pairs_count: 100, filepath: '/tmp/export.jsonl' })
    render(<ConversationDataSection />)
    await screen.findByText('20')
    fireEvent.click(screen.getByText('Export 100 Pairs for Training'))
    await vi.waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Exported 100 pairs to /tmp/export.jsonl', 'success')
    })
  })

  it('shows error toast on export failure', async () => {
    mockTrainingJobs.exportFeedbackPairs.mockRejectedValue(new Error('API down'))
    render(<ConversationDataSection />)
    await screen.findByText('20')
    fireEvent.click(screen.getByText('Export 100 Pairs for Training'))
    await vi.waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Something went wrong exporting', 'error')
    })
  })

  it('changes export button text while exporting', async () => {
    mockTrainingJobs.exportFeedbackPairs.mockReturnValue(new Promise(() => {}))
    render(<ConversationDataSection />)
    await screen.findByText('20')
    fireEvent.click(screen.getByText('Export 100 Pairs for Training'))
    expect(screen.getByText('Exporting...')).toBeDefined()
  })
})
