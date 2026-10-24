import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const { mockPush, bootstrapState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  bootstrapState: { payload: null as any },
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: { list: vi.fn(), load: vi.fn() },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { list: vi.fn(), switch: vi.fn() },
}))

vi.mock('@/contexts/BootstrapContext', () => ({
  useBootstrap: () => ({
    data: bootstrapState.payload,
    loading: bootstrapState.payload === null,
    error: null,
    refresh: vi.fn(),
  }),
}))

vi.mock('@/lib/store', () => ({
  useSettings: () => ({ theme: 'light' }),
  useUpdateSettings: () => vi.fn(),
}))

import { CommandPalette } from './CommandPalette'

describe('CommandPalette', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    bootstrapState.payload = null
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
    // selectedIdx starts at 0; ArrowDown once → index 1 = Training
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(mockPush).toHaveBeenCalledWith('/training')
  })

  it('lists recent sessions from the bootstrap payload on mount', () => {
    bootstrapState.payload = {
      members: { sessions: [{ id: 's1', name: 'My Chat' }], models: null, souls: null },
      errors: {},
    }
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    expect(screen.getByText('My Chat')).toBeDefined()
  })
})
