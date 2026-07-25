import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockPush, mockSessionList } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockSessionList: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {},
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: { list: mockSessionList },
}))

import { CommandPalette } from './CommandPalette'

describe('CommandPalette', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockSessionList.mockResolvedValue([])
  })
  afterEach(cleanup)

  it('returns null when closed', () => {
    const { container } = render(<CommandPalette />)
    expect(container.innerHTML).toBe('')
  })

  it('opens on Cmd+K', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.getByPlaceholderText('Search conversations, models, pages...')).toBeDefined()
  })

  it('closes on Escape', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Search conversations, models, pages...')).toBeNull()
  })

  it('closes on backdrop click', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const backdrop = document.querySelector('.fixed.inset-0')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop!)
    expect(screen.queryByPlaceholderText('Search conversations, models, pages...')).toBeNull()
  })

  it('filters actions by query', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search conversations, models, pages...')
    fireEvent.change(input, { target: { value: 'New' } })
    expect(screen.getByText('New Chat')).toBeDefined()
    expect(screen.queryByText('Export Chat')).toBeNull()
  })

  it('shows "No results" for unmatched query', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search conversations, models, pages...')
    fireEvent.change(input, { target: { value: 'zzzznotfound' } })
    expect(screen.getByText(/No results/)).toBeDefined()
  })

  it('navigates with arrow keys and enters', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search conversations, models, pages...')
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    // Second ArrowDown selects "Search Conversations" (index 1), Enter runs it
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('loads recent sessions on mount', () => {
    mockSessionList.mockResolvedValue([{ id: 's1', name: 'My Chat' }])
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(mockSessionList).toHaveBeenCalled()
  })
})
