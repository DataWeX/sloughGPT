import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ModelCacheCard from './ModelCacheCard'

describe('ModelCacheCard', () => {
  afterEach(cleanup)

  const base = {
    cacheUsage: { total_gb: 3.5, model_count: 4 },
    health: { status: 'healthy', model_loaded: true, model_type: 'gpt2', summary: 'ok' },
    onRefresh: vi.fn(),
  }

  it('renders cache stats', () => {
    render(<ModelCacheCard {...base} />)
    expect(screen.getByText('4')).toBeDefined()
    expect(screen.getByText('3.5 GB')).toBeDefined()
    expect(screen.getByText('GPT 2')).toBeDefined()
  })

  it('shows skeleton loading state when cacheUsage is null', () => {
    render(<ModelCacheCard {...base} cacheUsage={null} />)
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows None when model not loaded', () => {
    render(<ModelCacheCard {...base} health={{ status: 'error', model_loaded: false, model_type: '', summary: 'error' }} />)
    expect(screen.getByText('None')).toBeDefined()
  })

  it('shows offline health', () => {
    render(<ModelCacheCard {...base} health={null} />)
    expect(screen.getByText('None')).toBeDefined()
  })

  it('renders model count and total size', () => {
    render(<ModelCacheCard {...base} />)
    expect(screen.getByText('4')).toBeDefined()
    expect(screen.getByText('3.5 GB')).toBeDefined()
  })

  it('renders current model type', () => {
    render(<ModelCacheCard {...base} />)
    expect(screen.getByText('GPT 2')).toBeDefined()
  })

  it('renders card structure', () => {
    const { container } = render(<ModelCacheCard {...base} />)
    expect(container.querySelector('.rounded-lg')).toBeTruthy()
  })
})
