import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

vi.mock('@/lib/controllers', () => ({
  multimodalController: {
    getCapabilities: vi.fn().mockResolvedValue(null),
    getTrainingReport: vi.fn().mockResolvedValue(null),
    getTrainingStatus: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: vi.fn().mockReturnValue('Error'),
}))

vi.mock('@/lib/http-client', () => ({
  apiPost: vi.fn(),
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}))

vi.mock('next/dynamic', () => {
  const React = require('react')
  return {
    __esModule: true,
    default: () => (props: Record<string, unknown>) => React.createElement('div', { 'data-testid': 'dynamic' }),
  }
})

vi.mock('@/components/multimodal/CapabilitiesCard', () => ({ default: () => <div data-testid="capabilities-card" /> }))
vi.mock('@/components/multimodal/TrainingCard', () => ({ default: () => <div data-testid="training-card" /> }))
vi.mock('@/components/multimodal/ImageTrainingCard', () => ({ default: () => <div data-testid="image-training-card" /> }))
vi.mock('@/components/multimodal/BatchTrainingCard', () => ({ default: () => <div data-testid="batch-training-card" /> }))
vi.mock('@/components/multimodal/VisualDatasetCard', () => ({ default: () => <div data-testid="visual-dataset-card" /> }))
vi.mock('@/components/multimodal/DPOCard', () => ({ default: () => <div data-testid="dpo-card" /> }))
vi.mock('@/components/multimodal/ImageGenerationCard', () => ({ default: () => <div data-testid="image-generation-card" /> }))
vi.mock('@/components/multimodal/AudioCard', () => ({ default: () => <div data-testid="audio-card" /> }))

import MultimodalPage from './page'

describe('MultimodalPage', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('renders without crashing', async () => {
    render(<MultimodalPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('renders page header', async () => {
    render(<MultimodalPage />)
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
    await act(async () => {})
  })
})
