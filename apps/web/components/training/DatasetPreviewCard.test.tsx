// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { DatasetPreviewCard } from './DatasetPreviewCard'

vi.mock('@/lib/controllers', () => ({
  datasetController: {
    preview: vi.fn(),
  },
}))

import { datasetController } from '@/lib/controllers'
const mockPreview = vi.mocked(datasetController.preview)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('DatasetPreviewCard', () => {
  it('returns null when no datasetId', () => {
    const { container } = render(<DatasetPreviewCard datasetId={null} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows loading skeleton', () => {
    mockPreview.mockReturnValue(new Promise(() => {}))
    render(<DatasetPreviewCard datasetId="test-ds" />)
    expect(screen.getByText('Dataset preview')).toBeDefined()
  })

  it('shows preview data after load', async () => {
    mockPreview.mockResolvedValue({
      dataset_id: 'test-ds',
      samples: [
        { path: 'a.txt', language: 'en', content: 'Hello world', size: 11 },
        { path: 'b.txt', language: 'en', content: 'Second line', size: 12 },
      ],
      total_samples: 100,
      total_chars: 5000,
      languages: { en: 90, es: 10 },
    })

    render(<DatasetPreviewCard datasetId="test-ds" />)

    await waitFor(() => {
      expect(screen.getByText('100 samples')).toBeDefined()
    })
    expect(screen.getByText('5.0K chars')).toBeDefined()
    expect(screen.getByText('Hello world')).toBeDefined()
  })

  it('shows error state', async () => {
    mockPreview.mockRejectedValue(new Error('not found'))

    render(<DatasetPreviewCard datasetId="bad-ds" />)

    await waitFor(() => {
      expect(screen.getByText('Could not load preview')).toBeDefined()
    })
  })

  it('shows char count per sample', async () => {
    mockPreview.mockResolvedValue({
      dataset_id: 'ds',
      samples: [
        { path: 'a.txt', language: 'en', content: 'Short', size: 5 },
      ],
      total_samples: 10,
      total_chars: 500,
      languages: {},
    })

    render(<DatasetPreviewCard datasetId="ds" />)

    await waitFor(() => {
      expect(screen.getByText('~50 chars/sample')).toBeDefined()
    })
  })
})
