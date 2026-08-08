// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve({ data: { status: 'not_started', history: [] } }) }))

import { SelfTrainCard } from './SelfTrainCard'

describe('SelfTrainCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve({ data: { status: 'not_started', history: [] } }) }))
  })

  it('renders card title', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getByText('Self-Train')).toBeTruthy())
  })

  it('renders model input', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByPlaceholderText(/model/i).length).toBeGreaterThanOrEqual(1))
  })

  it('renders start button when idle', async () => {
    render(<SelfTrainCard />)
    await waitFor(() => expect(screen.getAllByText('Start').length).toBeGreaterThanOrEqual(1))
  })
})
