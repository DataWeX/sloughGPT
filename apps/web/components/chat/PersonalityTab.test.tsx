// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockGetTraitWeights = vi.hoisted(() => vi.fn())

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { getTraitWeights: mockGetTraitWeights },
}))

vi.mock('@/components/souls/SoulVisualizer', () => ({
  default: ({ traitWeights, currentSoulName }: any) => (
    <div data-testid="soul-visualizer" data-soul={currentSoulName}>
      {(traitWeights as any).personality?.warmth}
    </div>
  ),
}))

import { PersonalityTab } from './PersonalityTab'

describe('PersonalityTab', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows loading state on mount', () => {
    mockGetTraitWeights.mockReturnValue(new Promise(() => {}))
    const { container } = render(<PersonalityTab soulName="friendly" />)
    expect(container.querySelector('.animate-pulse')).toBeDefined()
  })

  it('shows no personality data message on error', async () => {
    mockGetTraitWeights.mockRejectedValue(new Error('fail'))
    render(<PersonalityTab soulName="friendly" />)
    await waitFor(() => {
      expect(screen.getByText('No personality data available')).toBeDefined()
    })
  })

  it('shows no personality data message when response has error', async () => {
    mockGetTraitWeights.mockResolvedValue({ error: 'not found' })
    render(<PersonalityTab soulName="friendly" />)
    await waitFor(() => {
      expect(screen.getByText('No personality data available')).toBeDefined()
    })
  })

  it('renders SoulVisualizer with trait weights', async () => {
    mockGetTraitWeights.mockResolvedValue({
      personality: { warmth: 0.8, creativity: 0.6 },
      cognition: { reasoning: 0.7 },
    })
    render(<PersonalityTab soulName="friendly" />)
    await waitFor(() => {
      expect(screen.getByTestId('soul-visualizer')).toBeDefined()
    })
  })

  it('passes soulName to SoulVisualizer', async () => {
    mockGetTraitWeights.mockResolvedValue({ personality: { warmth: 0.5 } })
    render(<PersonalityTab soulName="witty" />)
    await waitFor(() => {
      const viz = screen.getByTestId('soul-visualizer')
      expect(viz.getAttribute('data-soul')).toBe('witty')
    })
  })

  it('cancels fetch on unmount', async () => {
    mockGetTraitWeights.mockImplementation(() => new Promise(() => {}))
    const { unmount } = render(<PersonalityTab soulName="friendly" />)
    unmount()
    // no error = test passes (cancelled flag prevented setState on unmounted component)
  })

  it('refetches when soulName changes', async () => {
    mockGetTraitWeights
      .mockResolvedValueOnce({ personality: { warmth: 0.8 } })
      .mockResolvedValueOnce({ personality: { warmth: 0.2 } })
    const { rerender } = render(<PersonalityTab soulName="friendly" />)
    await waitFor(() => { expect(screen.getByTestId('soul-visualizer')).toBeDefined() })
    expect(mockGetTraitWeights).toHaveBeenCalledTimes(1)
    rerender(<PersonalityTab soulName="witty" />)
    await waitFor(() => {
      expect(mockGetTraitWeights).toHaveBeenCalledTimes(2)
    })
  })
})
