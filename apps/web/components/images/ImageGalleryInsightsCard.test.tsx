// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ImageGalleryInsightsCard } from './ImageGalleryInsightsCard'

afterEach(() => { cleanup() })

const now = Date.now() / 1000
const makeImage = (hoursAgo: number) => ({
  id: `img-${hoursAgo}`,
  path: `/images/${hoursAgo}.png`,
  created: now - hoursAgo * 3600,
})

const styles = [
  { key: 'realistic', name: 'Realistic' },
  { key: 'abstract', name: 'Abstract' },
  { key: 'cartoon', name: 'Cartoon' },
]

describe('ImageGalleryInsightsCard', () => {
  it('returns null for empty gallery', () => {
    const { container } = render(<ImageGalleryInsightsCard gallery={[]} styles={styles} />)
    expect(container.querySelector('[data-testid="image-gallery-insights"]')).toBeNull()
  })

  it('renders card when gallery has items', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1), makeImage(5)]} styles={styles} />)
    expect(screen.getAllByTestId('image-gallery-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Gallery Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total count', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1), makeImage(5), makeImage(10)]} styles={styles} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows last 24h count', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1), makeImage(5), makeImage(50)]} styles={styles} />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('shows styles count', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1)]} styles={styles} />)
    expect(screen.getAllByText(/3 styles available/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows singular style when one style', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1)]} styles={[styles[0]]} />)
    expect(screen.getAllByText(/1 style available/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows last 7d count', () => {
    render(<ImageGalleryInsightsCard gallery={[makeImage(1), makeImage(50), makeImage(200)]} styles={styles} />)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })
})
