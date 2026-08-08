import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockPush, mockSessionList, mockModelList } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockSessionList: vi.fn(),
  mockModelList: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: { list: mockModelList },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: { list: mockSessionList },
}))

vi.mock('@/lib/store', () => ({
  useSettings: () => ({ theme: 'light' }),
  useUpdateSettings: () => vi.fn(),
}))

import { CommandPalette } from './CommandPalette'

describe('CommandPalette', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockSessionList.mockResolvedValue([])
    mockModelList.mockResolvedValue([])
  })
  afterEach(cleanup)

  it('returns null when closed', () => {
    const { container } = render(<CommandPalette />)
    expect(container.innerHTML).toBe('')
  })

  it('opens on Cmd+K', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.getByPlaceholderText('Search pages, models, actions...')).toBeDefined()
  })

  it('closes on Escape', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Search pages, models, actions...')).toBeNull()
  })

  it('closes on backdrop click', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const backdrop = document.querySelector('.fixed.inset-0')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop!)
    expect(screen.queryByPlaceholderText('Search pages, models, actions...')).toBeNull()
  })

  it('filters actions by query', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    fireEvent.change(input, { target: { value: 'New' } })
    expect(screen.getByText('New Chat')).toBeDefined()
    expect(screen.queryByText('Export Chat')).toBeNull()
  })

  it('shows "No results" for unmatched query', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    fireEvent.change(input, { target: { value: 'zzzznotfound' } })
    expect(screen.getByText(/No results/)).toBeDefined()
  })

  it('navigates with arrow keys and enters', () => {
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    // selectedIdx starts at 0; ArrowDown twice → index 2 = Training
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(mockPush).toHaveBeenCalledWith('/training')
  })

  it('loads recent sessions on mount', () => {
    mockSessionList.mockResolvedValue([{ id: 's1', name: 'My Chat' }])
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(mockSessionList).toHaveBeenCalled()
  })
})
