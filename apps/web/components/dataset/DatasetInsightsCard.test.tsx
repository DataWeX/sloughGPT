// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { DatasetInsightsCard } from './DatasetInsightsCard'
import type { DatasetPreview } from '@/lib/dataset-controller'

afterEach(() => { cleanup() })

function makePreview(overrides: Partial<DatasetPreview> = {}): DatasetPreview {
  return {
    dataset_id: 'test',
    samples: [
      { path: 'a.txt', language: 'en', content: 'Hello world this is a test sentence with enough words', size: 100 },
      { path: 'b.txt', language: 'en', content: 'Another sample with different content for variety testing purposes', size: 200 },
      { path: 'c.txt', language: 'es', content: 'Hola mundo esto es una prueba', size: 150 },
    ],
    total_samples: 3,
    total_chars: 450,
    languages: { en: 2, es: 1 },
    ...overrides,
  }
}

describe('DatasetInsightsCard', () => {
  it('renders empty state for empty preview', () => {
    render(<DatasetInsightsCard preview={makePreview({ samples: [] })} />)
    expect(screen.getAllByText('No dataset insights available').length).toBeGreaterThanOrEqual(1)
  })

  it('renders empty state for null preview', () => {
    render(<DatasetInsightsCard preview={null} />)
    expect(screen.getAllByText('No dataset insights available').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading skeleton', () => {
    render(<DatasetInsightsCard preview={null} loading={true} />)
    expect(screen.getAllByTestId('dataset-insights').length).toBeGreaterThanOrEqual(1)
  })

  it('renders insights card with stats', () => {
    render(<DatasetInsightsCard preview={makePreview()} />)
    expect(screen.getAllByTestId('dataset-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average words and length', () => {
    render(<DatasetInsightsCard preview={makePreview()} />)
    expect(screen.getAllByText('Avg Words').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Avg Length').length).toBeGreaterThanOrEqual(1)
  })

  it('shows language badges', () => {
    render(<DatasetInsightsCard preview={makePreview()} />)
    expect(screen.getAllByText(/en/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/es/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows word range', () => {
    render(<DatasetInsightsCard preview={makePreview()} />)
    expect(screen.getAllByText('Word range').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty samples warning when present', () => {
    render(<DatasetInsightsCard preview={makePreview({
      samples: [
        { path: 'a.txt', language: 'en', content: '', size: 0 },
        { path: 'b.txt', language: 'en', content: 'Has content', size: 11 },
      ],
      languages: { en: 2 },
    })} />)
    expect(screen.getAllByText('Empty samples').length).toBeGreaterThanOrEqual(1)
  })

  it('shows short samples warning when present', () => {
    render(<DatasetInsightsCard preview={makePreview({
      samples: [
        { path: 'a.txt', language: 'en', content: 'hi', size: 2 },
      ],
      languages: { en: 1 },
    })} />)
    expect(screen.getAllByText(/Very short/).length).toBeGreaterThanOrEqual(1)
  })

  it('computes diversity score', () => {
    render(<DatasetInsightsCard preview={makePreview()} />)
    expect(screen.getAllByText('Diversity').length).toBeGreaterThanOrEqual(1)
  })
})
