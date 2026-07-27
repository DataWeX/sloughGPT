import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { EvalReportCard } from './EvalReportCard'
import { trainingController } from '@/lib/controllers'

vi.mock('@/lib/controllers', () => ({
  trainingController: {
    getEvalHistory: vi.fn(),
  },
}))

const mockGetEvalHistory = vi.mocked(trainingController.getEvalHistory)

beforeEach(() => {
  mockGetEvalHistory.mockReset()
})

const mkBaseline = () => ({
  timestamp: '2026-05-25T14:24:28',
  adapter_path: null,
  prompts: 8,
  references: 5,
  perplexity: 12.5,
  bleu: 8.9,
  avg_response_len: 40,
  inference_time_sec: 17,
  tokens_per_sec: 18.5,
  personality_score: 0.37,
})

describe('EvalReportCard', () => {
  it('renders loading skeleton', () => {
    mockGetEvalHistory.mockReturnValue(new Promise(() => {}))
    render(<EvalReportCard />)
    expect(screen.getByText('Evaluation History')).toBeDefined()
  })

  it('shows empty state when no eval history', async () => {
    mockGetEvalHistory.mockResolvedValue({ results: [] })
    render(<EvalReportCard />)
    await waitFor(() => {
      expect(screen.getByText(/No evaluation reports yet/)).toBeDefined()
    })
  })

  it('renders eval entries with verdict', async () => {
    mockGetEvalHistory.mockResolvedValue({
      results: [
        {
          timestamp: '2026-05-25T14:24:28',
          baseline: mkBaseline(),
          with_adapter: { ...mkBaseline(), perplexity: 11.2, bleu: 8.3, tokens_per_sec: 1.9 },
          delta: { perplexity_delta: -0.104, bleu_delta: -0.068, throughput_delta: -0.897, verdict: 'improved' },
        },
      ],
    })
    render(<EvalReportCard />)
    await waitFor(() => {
      expect(screen.getByText(/Improved/)).toBeDefined()
      expect(screen.getByText(/PPL: 12.5/)).toBeDefined()
      expect(screen.getByText(/BLEU: 8.9/)).toBeDefined()
    })
  })

  it('shows expand button when more than 3 entries', async () => {
    const entries = Array.from({ length: 5 }, (_, i) => ({
      timestamp: `2026-05-2${i}T14:00:00`,
      baseline: mkBaseline(),
    }))
    mockGetEvalHistory.mockResolvedValue({ results: entries })
    render(<EvalReportCard />)
    await waitFor(() => {
      expect(screen.getByText('Show all 5')).toBeDefined()
    })
  })
})
