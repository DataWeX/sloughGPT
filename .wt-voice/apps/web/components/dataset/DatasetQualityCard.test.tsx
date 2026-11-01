// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { DatasetQualityCard } from './DatasetQualityCard'

const { mockPreview } = vi.hoisted(() => ({
  mockPreview: vi.fn(),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    preview: (...args: any[]) => mockPreview(...args),
  },
}))

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('DatasetQualityCard', () => {
  it('shows loading state initially', async () => {
    mockPreview.mockReturnValue(new Promise(() => {}))
    render(<DatasetQualityCard datasetId="test" />)
    expect(screen.getAllByTestId('dataset-quality').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Analyzing data quality...').length).toBeGreaterThanOrEqual(1)
  })

  it('renders quality metrics after load', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 100,
      samples: Array.from({ length: 100 }, (_, i) => ({
        content: `Line ${i} with some content here for testing`,
        path: `file${i}.txt`,
        language: 'en',
        size: 40,
      })),
      total_chars: 4000,
      languages: { en: 100 },
    })
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Total lines').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('100').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Avg chars').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Good badge when no issues', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 10,
      samples: Array.from({ length: 10 }, (_, i) => ({
        content: `This is a normal length line of text for line ${i}`,
        path: `file${i}.txt`,
        language: 'en',
        size: 50,
      })),
      total_chars: 500,
      languages: { en: 10 },
    })
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Good').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows Fair badge with 1-2 issues', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 10,
      samples: Array.from({ length: 10 }, (_, i) => ({
        content: i < 2 ? '' : `Normal content for line ${i} with enough text`,
        path: `file${i}.txt`,
        language: 'en',
        size: 40,
      })),
      total_chars: 360,
      languages: { en: 10 },
    })
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Fair').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows duplicate count when duplicates exist', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 10,
      samples: Array.from({ length: 10 }, (_, i) => ({
        content: i < 3 ? 'duplicate line' : `Unique content for line ${i}`,
        path: `file${i}.txt`,
        language: 'en',
        size: 30,
      })),
      total_chars: 300,
      languages: { en: 10 },
    })
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Duplicates').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows empty line count', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 5,
      samples: [
        { content: 'Normal line', path: 'a.txt', language: 'en', size: 11 },
        { content: '', path: 'b.txt', language: 'en', size: 0 },
        { content: '', path: 'c.txt', language: 'en', size: 0 },
        { content: 'Another line', path: 'd.txt', language: 'en', size: 12 },
        { content: 'Third line', path: 'e.txt', language: 'en', size: 10 },
      ],
      total_chars: 33,
      languages: { en: 5 },
    })
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Empty lines').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows placeholder when no preview data', async () => {
    mockPreview.mockResolvedValue(null)
    render(<DatasetQualityCard datasetId="test" />)
    await waitFor(() => {
      expect(screen.getAllByText('Preview data to analyze quality.').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('calls preview on mount', async () => {
    mockPreview.mockResolvedValue({
      total_samples: 0,
      samples: [],
      total_chars: 0,
      languages: {},
    })
    render(<DatasetQualityCard datasetId="my-dataset" />)
    await waitFor(() => {
      expect(mockPreview).toHaveBeenCalledWith('my-dataset', 200)
    })
  })
})
