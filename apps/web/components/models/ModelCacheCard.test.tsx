import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ModelCacheCard from './ModelCacheCard'

describe('ModelCacheCard', () => {
  afterEach(cleanup)

  const base = {
    cacheUsage: { total_gb: 3.5, model_count: 4 },
    health: { model_loaded: true, model_type: 'gpt2' },
    onRefresh: vi.fn(),
  }

  it('renders cache stats', () => {
    render(<ModelCacheCard {...base} />)
    expect(screen.getByText('4')).toBeDefined()
    expect(screen.getByText('3.5 GB')).toBeDefined()
    expect(screen.getByText('GPT 2')).toBeDefined()
  })

  it('shows loading state when cacheUsage is null', () => {
    render(<ModelCacheCard {...base} cacheUsage={null} />)
    expect(screen.getByText('Loading cache stats...')).toBeDefined()
  })

  it('shows None when model not loaded', () => {
    render(<ModelCacheCard {...base} health={{ model_loaded: false }} />)
    expect(screen.getByText('None')).toBeDefined()
  })

  it('shows offline health', () => {
    render(<ModelCacheCard {...base} health="offline" />)
    expect(screen.getByText('None')).toBeDefined()
  })
})
