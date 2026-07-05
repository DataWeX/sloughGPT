import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { TokenBadge, SegBar, SampleTable } from './TokenizerComponents'

describe('TokenBadge', () => {
  afterEach(cleanup)

  it('renders token text', () => {
    render(<TokenBadge token="hello" />)
    expect(screen.getByText('hello')).toBeDefined()
  })

  it('shows special styling for special tokens', () => {
    const { container } = render(<TokenBadge token="<|endoftext|>" isSpecial />)
    const badge = container.querySelector('[class*="bg-primary"]')
    expect(badge).toBeDefined()
  })

  it('renders space as special character', () => {
    render(<TokenBadge token=" " />)
    expect(screen.getByText('␣')).toBeDefined()
  })

  it('renders newline as special character', () => {
    render(<TokenBadge token={'\n'} />)
    expect(screen.getByText('↵')).toBeDefined()
  })
})

describe('SegBar', () => {
  afterEach(cleanup)

  it('renders percentage and label', () => {
    render(<SegBar pct={75} label="training" />)
    expect(screen.getByText('75%')).toBeDefined()
    expect(screen.getByText('training')).toBeDefined()
  })

  it('formats percentage to one decimal properly', () => {
    render(<SegBar pct={33.33} label="test" />)
    expect(screen.getByText('33%')).toBeDefined()
  })
})

const { mockGetSamples } = vi.hoisted(() => ({
  mockGetSamples: vi.fn(),
}))

vi.mock('@/lib/tokenizer-controller', () => ({
  tokenizerController: {
    getSamples: mockGetSamples,
  },
}))

describe('SampleTable', () => {
  const mockSamples = [
    { word: 'hello', ids: [1, 2], tokens: ['hel', 'lo'], count: 2 },
    { word: 'world', ids: [3, 4, 5], tokens: ['wor', 'l', 'd'], count: 3 },
  ]

  beforeEach(() => {
    mockGetSamples.mockResolvedValue({ samples: mockSamples })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders sample words after fetch', async () => {
    render(<SampleTable />)
    const hello = await screen.findByText('hello')
    expect(hello).toBeDefined()
  })

  it('renders token IDs for samples', async () => {
    render(<SampleTable />)
    const ids = await screen.findByText('[1, 2]')
    expect(ids).toBeDefined()
  })

  it('renders empty state when no samples', async () => {
    mockGetSamples.mockResolvedValue({ samples: [] })
    render(<SampleTable />)
    const empty = await screen.findByText(/No samples available/)
    expect(empty).toBeDefined()
  })
})
