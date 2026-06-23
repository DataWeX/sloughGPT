// @vitest-environment jsdom
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

import { SelfTrainToggle } from './SelfTrainToggle'

describe('SelfTrainToggle', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ running: false, steps: 42 }),
    }))
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders status after mount', async () => {
    render(<SelfTrainToggle />)
    await waitFor(() => expect(screen.getAllByText(/42/).length >= 1).toBe(true))
  })

  it('shows Ready when not running', async () => {
    render(<SelfTrainToggle />)
    await waitFor(() => expect(screen.getAllByText('Ready').length >= 1).toBe(true))
    expect(screen.getAllByText('Start').length >= 1).toBe(true)
  })
})
