// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

import ImportResultModal from './ImportResultModal'

describe('ImportResultModal', () => {
  afterEach(cleanup)

  const result = { ok: 5, fail: 1, names: ['chat1', 'chat2', 'chat3'] }

  it('renders import stats', () => {
    render(<ImportResultModal result={result} onClose={vi.fn()} />)
    expect(screen.getByText(/5 imported/)).toBeDefined()
    expect(screen.getByText(/1 failed/)).toBeDefined()
  })

  it('renders imported names', () => {
    render(<ImportResultModal result={result} onClose={vi.fn()} />)
    expect(screen.getByText('chat1')).toBeDefined()
    expect(screen.getByText('chat2')).toBeDefined()
    expect(screen.getByText('chat3')).toBeDefined()
  })

  it('calls onClose when clicking backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(<ImportResultModal result={result} onClose={onClose} />)
    const backdrop = container.querySelector('.fixed')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop!)
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('calls onClose when clicking Close button', () => {
    const onClose = vi.fn()
    render(<ImportResultModal result={result} onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalledOnce()
  })
})
