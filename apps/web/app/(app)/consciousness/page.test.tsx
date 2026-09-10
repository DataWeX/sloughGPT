/**
 * Tests for the Consciousness settings page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ConsciousnessPage from './page'

const mockStatus = {
  enabled: true,
  level: 1,
  current_qualia: {
    valence: 0.5,
    arousal: 0.3,
    novelty: 0.6,
    coherence: 0.7,
    salience: 0.4,
    certainty: 0.3,
    complexity: 0.5,
  },
  beliefs: { helpful: 0.9, accurate: 0.7 },
  episodes: 10,
  narrative: 'Test narrative',
  training: {
    is_training: false,
    total_pairs: 15,
    current_epoch: 0,
    loss: 0.0,
  },
}

const mockEval = {
  overall_score: 75,
  metrics: {
    coherence: { score: 80, weight: 0.2, details: 'Good' },
    growth: { score: 70, weight: 0.15, details: 'OK' },
  },
}

describe('ConsciousnessPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url.includes('/status')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: mockStatus }) })
      }
      if (url.includes('/evaluate')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: mockEval }) })
      }
      if (url.includes('/reflect')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: { reflection: 'I am reflecting.' } }) })
      }
      if (url.includes('/config')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: { level: 2, enabled: true } }) })
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ data: {} }) })
    }))
  })

  it('renders the page title', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Consciousness Level').length).toBeGreaterThan(0)
    })
  })

  it('displays qualia metrics', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('valence').length).toBeGreaterThan(0)
      expect(screen.getAllByText('coherence').length).toBeGreaterThan(0)
    })
  })

  it('displays beliefs', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Self-Model Beliefs').length).toBeGreaterThan(0)
      expect(screen.getAllByText('helpful').length).toBeGreaterThan(0)
      expect(screen.getAllByText('90%').length).toBeGreaterThan(0)
    })
  })

  it('shows reflect button', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Reflect').length).toBeGreaterThan(0)
    })
  })

  it('calls reflect endpoint on click', async () => {
    const user = userEvent.setup()
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Reflect').length).toBeGreaterThan(0)
    })
    const reflectBtns = screen.getAllByText('Reflect')
    await user.click(reflectBtns[0])
    await waitFor(() => {
      expect(screen.getAllByText('I am reflecting.').length).toBeGreaterThan(0)
    })
  })

  it('displays evaluation report', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Evaluation Report').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Score: 75/100').length).toBeGreaterThan(0)
    })
  })

  it('shows training status', async () => {
    render(<ConsciousnessPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Training Status').length).toBeGreaterThan(0)
      expect(screen.getAllByText('15').length).toBeGreaterThan(0)
    })
  })
})
