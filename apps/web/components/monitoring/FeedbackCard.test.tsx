import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockApiPost, mockLoggerError } = vi.hoisted(() => ({
  mockApiPost: vi.fn(),
  mockLoggerError: vi.fn(),
}))

vi.mock('@/lib/http-client', () => ({
  apiPost: mockApiPost,
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { error: mockLoggerError },
}))

import { FeedbackCard } from './FeedbackCard'

const dpoStatus = {
  status: 'completed',
  last_run: '2026-08-06T10:00:00Z',
  accepted_count: 8,
  rejected_count: 2,
  result: { verdict: 'ok', perplexity_delta: -0.2, bleu_delta: 0.1, report_path: '/tmp/r' },
}

const visualStatus = { visual_loaded: true, training: { status: 'idle' } }

function renderCard(props: Partial<Parameters<typeof FeedbackCard>[0]> = {}) {
  const base = {
    dpoStatus: null,
    visualStatus: null,
    dpoRunning: false,
    onDpoRunningChange: vi.fn(),
    onRefresh: vi.fn(),
  }
  return render(<FeedbackCard {...base} {...props} />)
}

describe('FeedbackCard', () => {
  beforeEach(() => {
    mockApiPost.mockReset()
    mockLoggerError.mockReset()
    mockApiPost.mockResolvedValue({ ok: true })
  })

  afterEach(cleanup)

  it('renders nothing when both statuses are null', () => {
    const { container } = renderCard()
    expect(container.innerHTML).toBe('')
  })

  it('renders feedback status', () => {
    renderCard({ dpoStatus })
    expect(screen.getByText('completed')).toBeDefined()
  })

  it('renders vision status', () => {
    renderCard({ dpoStatus, visualStatus })
    expect(screen.getByText('Yes')).toBeDefined()
  })

  it('renders accepted and rejected counts', () => {
    renderCard({ dpoStatus })
    expect(screen.getByText('8')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('shows placeholder when vision status missing', () => {
    renderCard({ dpoStatus })
    expect(screen.getAllByText('...').length).toBeGreaterThan(0)
  })

  it('calls apiPost and onRefresh on run feedback', async () => {
    const onRefresh = vi.fn()
    renderCard({ dpoStatus, visualStatus, onRefresh })
    fireEvent.click(screen.getByText('Run feedback'))
    await waitFor(() => expect(mockApiPost).toHaveBeenCalledWith('/multimodal/dpo', {}))
    expect(onRefresh).toHaveBeenCalled()
  })

  it('toggles running state around the request', async () => {
    const onDpoRunningChange = vi.fn()
    let release: (v: unknown) => void = () => {}
    mockApiPost.mockImplementation(() => new Promise((res) => { release = res }))
    renderCard({ dpoStatus, visualStatus, onDpoRunningChange })
    fireEvent.click(screen.getByText('Run feedback'))
    expect(onDpoRunningChange).toHaveBeenCalledWith(true)
    release({ ok: true })
    await waitFor(() => expect(onDpoRunningChange).toHaveBeenCalledWith(false))
  })

  it('does not call onRefresh on error and logs it', async () => {
    mockApiPost.mockRejectedValue(new Error('boom'))
    const onRefresh = vi.fn()
    renderCard({ dpoStatus, visualStatus, onRefresh })
    fireEvent.click(screen.getByText('Run feedback'))
    await waitFor(() => expect(mockLoggerError).toHaveBeenCalled())
    expect(onRefresh).not.toHaveBeenCalled()
  })

  it('disables button while running', () => {
    renderCard({ dpoStatus, visualStatus, dpoRunning: true })
    expect(screen.getByText('Running...')).toBeDefined()
    expect(screen.getByRole('button')).toBeDisabled()
  })
})
