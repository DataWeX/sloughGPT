import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

import { SelfTrainProgress } from './SelfTrainProgress'

describe('SelfTrainProgress', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders nothing when no logs', async () => {
    mockFetch.mockResolvedValue({ json: () => Promise.resolve({ logs: [] }) })
    const { container } = render(<SelfTrainProgress />)
    expect(container.innerHTML).toBe('')
  })

  it('renders card with logs', async () => {
    mockFetch.mockResolvedValue({ json: () => Promise.resolve({ logs: ['Step 1: loss=1.2', 'Step 2: loss=0.8'] }) })
    render(<SelfTrainProgress />)
    const log1 = await screen.findByText('Step 1: loss=1.2')
    expect(log1).toBeDefined()
  })

  it('renders training log title', async () => {
    mockFetch.mockResolvedValue({ json: () => Promise.resolve({ logs: ['test log'] }) })
    render(<SelfTrainProgress />)
    const title = await screen.findByText('Training Log')
    expect(title).toBeDefined()
  })
})
