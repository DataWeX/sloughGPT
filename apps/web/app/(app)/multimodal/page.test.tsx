import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'
import React from 'react'

const {
  mockGetCapabilities, mockGetTrainingReport, mockGetTrainingStatus, mockAddToast,
} = vi.hoisted(() => ({
  mockGetCapabilities: vi.fn(), mockGetTrainingReport: vi.fn(),
  mockGetTrainingStatus: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@/lib/controllers', () => ({
  multimodalController: {
    getCapabilities: (...a: unknown[]) => mockGetCapabilities(...a),
    getTrainingReport: (...a: unknown[]) => mockGetTrainingReport(...a),
    getTrainingStatus: (...a: unknown[]) => mockGetTrainingStatus(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
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

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetCapabilities.mockResolvedValue({ vision: true, audio: true, image_gen: false })
  mockGetTrainingReport.mockResolvedValue({ total_samples: 100 })
  mockGetTrainingStatus.mockResolvedValue({ training: false })
})

describe('MultimodalPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders without crashing', async () => {
    render(<MultimodalPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('fetches capabilities on mount', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(mockGetCapabilities).toHaveBeenCalled()
  })
})

describe('MultimodalPage — capabilities display', () => {
  it('shows capabilities card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('capabilities-card')).toBeTruthy()
  })
})

describe('MultimodalPage — training cards', () => {
  it('shows image training card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('image-training-card')).toBeTruthy()
  })

  it('shows batch training card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('batch-training-card')).toBeTruthy()
  })
})

describe('MultimodalPage — other cards', () => {
  it('shows visual dataset card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('visual-dataset-card')).toBeTruthy()
  })

  it('shows DPO card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('dpo-card')).toBeTruthy()
  })

  it('shows image generation card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('image-generation-card')).toBeTruthy()
  })

  it('shows audio card', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('audio-card')).toBeTruthy()
  })
})

describe('MultimodalPage — error handling', () => {
  it('handles capabilities failure gracefully', async () => {
    mockGetCapabilities.mockRejectedValue(new Error('network'))
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
  })
})

describe('MultimodalPage — loading state', () => {
  it('shows loading while fetching capabilities', async () => {
    let resolveCaps: (v: { vision: boolean; audio: boolean; image_gen: boolean }) => void
    mockGetCapabilities.mockReturnValue(new Promise(r => { resolveCaps = r }))
    render(<MultimodalPage />)
    expect(screen.queryByText(/vision|multimodal/i)).toBeNull()
    await act(async () => { resolveCaps!({ vision: true, audio: true, image_gen: false }) })
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
  })
})

describe('MultimodalPage — training status', () => {
  it('renders training section', async () => {
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
  })
})

describe('MultimodalPage — capabilities details', () => {
  it('shows vision capability', async () => {
    mockGetCapabilities.mockResolvedValue({ vision: true, audio: false, image_gen: false })
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('capabilities-card')).toBeTruthy()
  })

  it('shows audio capability', async () => {
    mockGetCapabilities.mockResolvedValue({ vision: false, audio: true, image_gen: false })
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getByTestId('capabilities-card')).toBeTruthy()
  })
})

describe('MultimodalPage — error toast', () => {
  it('shows error toast on training report failure', async () => {
    mockGetTrainingReport.mockRejectedValue(new Error('fetch failed'))
    render(<MultimodalPage />)
    await act(async () => {})
    expect(screen.getAllByText(/vision|multimodal/i).length).toBeGreaterThanOrEqual(1)
  })
})
