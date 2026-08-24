// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: vi.fn((...args: any[]) => args.join(' ')),
  Button: ({ children, onClick, variant, size, className, disabled, ...rest }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} data-size={size} disabled={disabled} {...rest}>{children}</button>
  ),
  Switch: ({ checked, onCheckedChange, disabled, 'aria-label': label, ...rest }: any) => (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onCheckedChange?.(!checked)}
      {...rest}
    />
  ),
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
  IconTrash: () => <span data-testid="icon-trash">trash</span>,
  IconSearch: () => <span data-testid="icon-search">search</span>,
  IconX: () => <span data-testid="icon-x">x</span>,
  IconClock: () => <span data-testid="icon-clock">clock</span>,
  IconEdit: () => <span data-testid="icon-edit">edit</span>,
}))

const hoisted = vi.hoisted(() => ({
  memoryController: {
    list: vi.fn(),
    stats: vi.fn(),
    delete: vi.fn(),
    clear: vi.fn(),
    setEnabled: vi.fn(),
    search: vi.fn(),
    store: vi.fn(),
    update: vi.fn(),
    consolidate: vi.fn(),
  },
  logger: { debug: vi.fn() },
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: hoisted.memoryController,
}))

vi.mock('@/lib/dev-log', () => ({
  logger: hoisted.logger,
}))

import { MemoryTab } from './MemoryTab'
import { publishMemoryEvent } from '@/lib/memory-events'

const item = (overrides: Partial<Record<string, unknown>> = {}) => ({
  id: 'm1',
  content: 'The user prefers espresso over drip coffee.',
  topic: 'preferences',
  source: 'auto',
  url: '',
  timestamp: 1700000000000,
  importance: 3,
  score: 0.8,
  ...overrides,
})

const statsResult = { enabled: true, total_facts: 1, topics: 1, visited_urls: 0 }

describe('MemoryTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hoisted.memoryController.list.mockResolvedValue({ items: [item()], total: 1 })
    hoisted.memoryController.stats.mockResolvedValue(statsResult)
    hoisted.memoryController.delete.mockResolvedValue({ deleted: 1 })
    hoisted.memoryController.clear.mockResolvedValue({ cleared: 1 })
    hoisted.memoryController.setEnabled.mockResolvedValue({ enabled: true })
    hoisted.memoryController.search.mockResolvedValue({ results: [], total: 0 })
    hoisted.memoryController.store.mockResolvedValue({ stored: true, content: 'stored', topic: 'manual', source: 'manual' })
    hoisted.memoryController.update.mockResolvedValue({ updated: 1, duplicate: false })
    hoisted.memoryController.consolidate.mockResolvedValue({ removed: 0, kept: 1, threshold: 0.9 })
  })
  afterEach(cleanup)

  it('renders fact count and memory items after load', async () => {
    render(<MemoryTab />)
    expect(await screen.findByText('1 fact')).toBeDefined()
    expect(screen.getByText(/prefers espresso over drip coffee/)).toBeDefined()
    expect(screen.getAllByText('preferences').length).toBeGreaterThanOrEqual(1)
    expect(hoisted.memoryController.list).toHaveBeenCalled()
    expect(hoisted.memoryController.stats).toHaveBeenCalled()
  })

  it('pluralizes the fact count', async () => {
    hoisted.memoryController.stats.mockResolvedValue({ ...statsResult, total_facts: 5 })
    hoisted.memoryController.list.mockResolvedValue({ items: [item(), item({ id: 'm2', content: 'Another fact.' })], total: 2 })
    render(<MemoryTab />)
    expect(await screen.findByText('5 facts')).toBeDefined()
  })

  it('falls back to list length when stats are missing', async () => {
    hoisted.memoryController.stats.mockResolvedValue(null)
    render(<MemoryTab />)
    expect(await screen.findByText('1 fact')).toBeDefined()
  })

  it('shows the empty state when nothing is remembered', async () => {
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    expect(await screen.findByText(/Nothing remembered yet/)).toBeDefined()
  })

  it('shows a Memory off state when disabled', async () => {
    hoisted.memoryController.stats.mockResolvedValue({ ...statsResult, enabled: false })
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    expect(await screen.findByText('Memory off')).toBeDefined()
    expect(screen.getByText(/Memory is off/)).toBeDefined()
  })

  it('deletes a memory item and removes it from the list', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Delete memory item'))
    await waitFor(() => expect(hoisted.memoryController.delete).toHaveBeenCalledWith('m1'))
    expect(screen.queryByText(/prefers espresso over drip coffee/)).toBeNull()
  })

  it('refetches when a delete fails', async () => {
    hoisted.memoryController.delete.mockRejectedValueOnce(new Error('boom'))
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Delete memory item'))
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2))
  })

  it('requires confirmation before clearing all memory', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Clear memory'))
    expect(await screen.findByText(/Clear all stored memory/)).toBeDefined()
    expect(hoisted.memoryController.clear).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Clear'))
    await waitFor(() => expect(hoisted.memoryController.clear).toHaveBeenCalled())
  })

  it('cancels the clear confirmation without clearing', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Clear memory'))
    await screen.findByText(/Clear all stored memory/)
    fireEvent.click(screen.getByText('Cancel'))
    await waitFor(() => expect(hoisted.memoryController.clear).not.toHaveBeenCalled())
    expect(screen.queryByText(/Clear all stored memory/)).toBeNull()
  })

  it('disables the clear button when there is nothing to clear', async () => {
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    await screen.findByText(/Nothing remembered yet/)
    const clearBtn = screen.getByLabelText('Clear memory') as HTMLButtonElement
    expect(clearBtn.disabled).toBe(true)
  })

  it('refreshes memory on refresh button click', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Refresh memory'))
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.memoryController.stats).toHaveBeenCalledTimes(2))
  })

  it('truncates long memory content', async () => {
    const long = 'x'.repeat(200)
    hoisted.memoryController.list.mockResolvedValue({ items: [item({ id: 'm9', content: long })], total: 1 })
    render(<MemoryTab />)
    expect(await screen.findByText(/x{160}…/)).toBeDefined()
  })

  it('shows the source badge and relative time on items', async () => {
    const recent = item({ id: 'm10', content: 'A fresh fact', source: 'manual', timestamp: Math.floor(Date.now() / 1000) - 60 })
    hoisted.memoryController.list.mockResolvedValue({ items: [recent], total: 1 })
    render(<MemoryTab />)
    await screen.findByText('A fresh fact')
    expect(screen.getAllByText('manual').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1m ago').length).toBeGreaterThanOrEqual(1)
  })

  it('omits the relative time when the item has no timestamp', async () => {
    hoisted.memoryController.list.mockResolvedValue({ items: [item({ timestamp: 0 })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    expect(screen.queryByText('Just now')).toBeNull()
  })

  it('shows a Show all toggle when there are more than 8 items', async () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ ...item({ id: `m${i}`, content: `Fact number ${i + 1}` }) }))
    hoisted.memoryController.list.mockResolvedValue({ items: many, total: 12 })
    render(<MemoryTab />)
    await screen.findByText('Fact number 1')
    expect(screen.queryByText('Fact number 12')).toBeNull()
    expect(screen.getAllByText('Show all 12').length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getAllByText('Show all 12')[0])
    expect(screen.getAllByText('Fact number 12').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Show fewer').length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getAllByText('Show fewer')[0])
    expect(screen.queryByText('Fact number 12')).toBeNull()
  })

  it('hides the Show all toggle when searching and restores the cap on clear', async () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ ...item({ id: `m${i}`, content: `Fact number ${i + 1}` }) }))
    hoisted.memoryController.list.mockResolvedValue({ items: many, total: 12 })
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'Only one search hit.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText('Fact number 1')
    fireEvent.click(screen.getAllByText('Show all 12')[0])
    expect(screen.getAllByText('Fact number 12').length).toBeGreaterThanOrEqual(1)

    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'hit' } })
    await screen.findByText('Only one search hit.')
    expect(screen.queryByText('Show all 12')).toBeNull()
    expect(screen.queryByText('Fact number 12')).toBeNull()

    fireEvent.click(screen.getByTestId('icon-x').closest('button')!)
    await screen.findByText('Fact number 1')
    expect(screen.queryByText('Fact number 12')).toBeNull()
    expect(screen.getAllByText('Show all 12').length).toBeGreaterThanOrEqual(1)
  })

  it('copies a fact to the clipboard when the content is clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByTitle('Click to copy'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('The user prefers espresso over drip coffee.'))
    expect(screen.getAllByText('Copied').length).toBeGreaterThanOrEqual(1)
  })

  it('copies a fact when activated via the keyboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.keyDown(screen.getByTitle('Click to copy'), { key: 'Enter' })
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('The user prefers espresso over drip coffee.'))
  })

  it('clears the Copied indicator after a short timeout', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    vi.useFakeTimers()
    try {
      render(<MemoryTab />)
      await act(async () => {})
      await act(async () => {})
      fireEvent.click(screen.getByTitle('Click to copy'))
      await act(async () => {})
      expect(screen.getAllByText('Copied').length).toBeGreaterThanOrEqual(1)
      act(() => { vi.advanceTimersByTime(1500) })
      expect(screen.queryByText('Copied')).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not crash when the clipboard is unavailable', async () => {
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByTitle('Click to copy'))
    await waitFor(() => expect(hoisted.logger.debug).toHaveBeenCalled())
    expect(screen.queryByText('Copied')).toBeNull()
  })

  it('stores a manual fact and highlights it in the list', async () => {
    const stored = item({ id: 'new1', content: 'The user is allergic to peanuts.', topic: 'health' })
    hoisted.memoryController.store.mockResolvedValue({ stored: true, content: stored.content, topic: 'health', source: 'manual' })
    hoisted.memoryController.list.mockResolvedValue({ items: [stored], total: 1 })
    render(<MemoryTab />)
    await screen.findByText('The user is allergic to peanuts.')
    fireEvent.click(screen.getByText('+ Store'))
    fireEvent.change(screen.getByLabelText('New memory fact'), { target: { value: 'The user is allergic to peanuts.' } })
    fireEvent.change(screen.getByLabelText('Memory fact topic'), { target: { value: 'health' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => expect(hoisted.memoryController.store).toHaveBeenCalledWith('The user is allergic to peanuts.', 'health'))
    const li = (await screen.findByText('The user is allergic to peanuts.')).closest('li')
    expect(li?.className).toContain('border-primary/60')
  })

  it('shows an inline error when the fact already exists', async () => {
    hoisted.memoryController.store.mockResolvedValue({ stored: false, content: 'dup', topic: 'manual', source: 'manual' })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('+ Store'))
    fireEvent.change(screen.getByLabelText('New memory fact'), { target: { value: 'Duplicate fact' } })
    fireEvent.click(screen.getByText('Save'))
    expect(await screen.findByText('Already remembered (or memory is disabled)')).toBeDefined()
    expect(hoisted.memoryController.list).toHaveBeenCalledTimes(1)
  })

  it('disables Save until content is provided', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('+ Store'))
    expect((screen.getByText('Save').closest('button') as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByLabelText('New memory fact'), { target: { value: 'Some fact' } })
    expect((screen.getByText('Save').closest('button') as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows the importance badge on items', async () => {
    hoisted.memoryController.list.mockResolvedValue({ items: [item({ importance: 0.8 })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    expect(screen.getAllByText('importance 0.8').length).toBeGreaterThanOrEqual(1)
  })

  it('filters the list by a topic chip and restores All', async () => {
    hoisted.memoryController.list.mockResolvedValue({
      items: [
        item({ id: 'm1' }),
        item({ id: 'm2', content: 'The Seine flows through Paris.', topic: 'geography' }),
      ],
      total: 2,
    })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByRole('button', { name: 'geography' }))
    expect(screen.queryByText(/prefers espresso over drip coffee/)).toBeNull()
    expect(screen.getAllByText('The Seine flows through Paris.').length).toBeGreaterThanOrEqual(1)
    fireEvent.click(screen.getByRole('button', { name: 'All' }))
    expect(screen.getByText(/prefers espresso over drip coffee/)).toBeDefined()
  })

  it('toggles a topic chip off by clicking it again', async () => {
    hoisted.memoryController.list.mockResolvedValue({
      items: [
        item({ id: 'm1' }),
        item({ id: 'm2', content: 'The Seine flows through Paris.', topic: 'geography' }),
      ],
      total: 2,
    })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByRole('button', { name: 'geography' }))
    expect(screen.queryByText(/prefers espresso over drip coffee/)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'geography' }))
    expect(screen.getByText(/prefers espresso over drip coffee/)).toBeDefined()
  })

  it('shows a topic empty state when the active topic is excluded by search', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'Only one search hit.', topic: 'manual' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByRole('button', { name: 'preferences' }))
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'hit' } })
    await screen.findByText('No memory in the "preferences" topic.')
  })

  it('caps the list at 8 within a filtered topic and counts the filtered set', async () => {
    const geo = Array.from({ length: 12 }, (_, i) => ({ ...item({ id: `g${i}`, content: `Geo fact ${i + 1}`, topic: 'geography' }) }))
    hoisted.memoryController.list.mockResolvedValue({ items: geo, total: 12 })
    render(<MemoryTab />)
    await screen.findByText('Geo fact 1')
    fireEvent.click(screen.getByRole('button', { name: 'geography' }))
    expect(screen.queryByText('Geo fact 12')).toBeNull()
    fireEvent.click(screen.getAllByText('Show all 12')[0])
    expect(screen.getAllByText('Geo fact 12').length).toBeGreaterThanOrEqual(1)
  })

  it('sorts items oldest first when toggled', async () => {
    hoisted.memoryController.list.mockResolvedValue({
      items: [
        item({ id: 'newer', content: 'Newer fact', timestamp: 3000 }),
        item({ id: 'older', content: 'Older fact', timestamp: 1000 }),
      ],
      total: 2,
    })
    render(<MemoryTab />)
    await screen.findByText('Newer fact')
    fireEvent.click(screen.getByLabelText('Toggle memory sort order'))
    const lis = screen.getAllByRole('listitem')
    expect(lis[0].textContent).toContain('Older fact')
    expect(lis[1].textContent).toContain('Newer fact')
  })

  it('shows the current sort state on the button', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    expect(screen.getByText('Newest')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Toggle memory sort order'))
    expect(screen.getByText('Oldest')).toBeDefined()
  })

  it('disables the sort toggle while searching', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'Only one search hit.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'hit' } })
    await screen.findByText('Only one search hit.')
    expect((screen.getByLabelText('Toggle memory sort order') as HTMLButtonElement).disabled).toBe(true)
  })

  it('searches memory after typing with debounce', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'The user prefers pour-over coffee.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'coffee' } })
    await waitFor(() => {
      expect(hoisted.memoryController.search).toHaveBeenCalledWith('coffee')
    }, { timeout: 2000 })
  })

  it('shows search results instead of the full list', async () => {
    hoisted.memoryController.list.mockResolvedValue({
      items: [item(), item({ id: 'm2', content: 'The Seine flows through Paris.' })],
      total: 2,
    })
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'The user prefers pour-over coffee.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'pour-over' } })
    await screen.findByText(/prefers pour-over coffee/)
    expect(screen.queryByText(/prefers espresso over drip coffee/)).toBeNull()
    expect(screen.queryByText(/The Seine flows through Paris/)).toBeNull()
  })

  it('offers a Clear search recovery action when nothing matches', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [], total: 0 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'nope' } })
    await screen.findByText('No memory matches that search.')
    fireEvent.click(screen.getByText('Clear search'))
    await waitFor(() => expect(screen.getByPlaceholderText('Search memory...')).toHaveValue(''))
    expect(await screen.findByText(/prefers espresso over drip coffee/)).toBeDefined()
  })

  it('clears the search from the input clear button', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'The user prefers pour-over coffee.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'pour-over' } })
    await screen.findByText(/prefers pour-over coffee/)
    const xBtn = screen.getByTestId('icon-x').closest('button')
    expect(xBtn).not.toBeNull()
    fireEvent.click(xBtn!)
    await waitFor(() => expect(screen.getByPlaceholderText('Search memory...')).toHaveValue(''))
    expect(await screen.findByText(/prefers espresso over drip coffee/)).toBeDefined()
  })

  it('does not show the search box while memory is off', async () => {
    hoisted.memoryController.stats.mockResolvedValue({ ...statsResult, enabled: false })
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    await screen.findByText('Memory off')
    expect(screen.queryByPlaceholderText('Search memory...')).toBeNull()
  })

  it('re-runs the active search when a memory event refreshes', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'The user prefers pour-over coffee.' })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'coffee' } })
    await waitFor(() => {
      expect(hoisted.memoryController.search).toHaveBeenCalledWith('coffee')
    }, { timeout: 2000 })
    publishMemoryEvent({ stored: true, fact: 'The user prefers pour-over coffee.' })
    await waitFor(() => {
      expect(hoisted.memoryController.search).toHaveBeenCalledTimes(2)
    }, { timeout: 2000 })
  })

  it('shows the Remember switch reflecting the enabled state', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    const sw = screen.getByLabelText('Toggle automatic memory') as HTMLButtonElement
    expect(sw.getAttribute('aria-checked')).toBe('true')
  })

  it('turns memory off from the panel switch', async () => {
    hoisted.memoryController.setEnabled.mockResolvedValue({ enabled: false })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Toggle automatic memory'))
    await waitFor(() => expect(hoisted.memoryController.setEnabled).toHaveBeenCalledWith(false))
  })

  it('turns memory back on from the panel switch when disabled', async () => {
    hoisted.memoryController.setEnabled.mockResolvedValue({ enabled: true })
    hoisted.memoryController.stats.mockResolvedValue({ ...statsResult, enabled: false })
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    await screen.findByText('Memory off')
    fireEvent.click(screen.getByLabelText('Toggle automatic memory'))
    await waitFor(() => expect(hoisted.memoryController.setEnabled).toHaveBeenCalledWith(true))
  })

  it('refetches when a stored memory event is published', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    publishMemoryEvent({ stored: true })
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(hoisted.memoryController.stats).toHaveBeenCalledTimes(2))
  })

  it('does not refetch on a non-stored memory event', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    publishMemoryEvent({ stored: false })
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(hoisted.memoryController.stats).toHaveBeenCalledTimes(1))
  })

  it('highlights the newly stored fact after a memory event', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    publishMemoryEvent({ stored: true, fact: 'The user prefers espresso over drip coffee.' })
    await waitFor(() => {
      const li = screen.getByText(/prefers espresso over drip coffee/).closest('li')
      expect(li?.className).toContain('border-primary/60')
    })
  })

  it('does not highlight when the published fact has no list match', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    publishMemoryEvent({ stored: true, fact: 'Some completely different fact.' })
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2))
    const li = screen.getByText(/prefers espresso over drip coffee/).closest('li')
    expect(li?.className).not.toContain('border-primary/60')
    expect(li?.className).toContain('border-border/40')
  })

  it('highlights only the first fact when a memory event carries a facts array', async () => {
    hoisted.memoryController.list.mockResolvedValue({
      items: [item(), item({ id: 'm2', content: 'The Seine flows through Paris.' })],
      total: 2,
    })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    publishMemoryEvent({ stored: true, fact: 'A.', facts: ['The user prefers espresso over drip coffee.', 'The Seine flows through Paris.'] })
    await waitFor(() => {
      const li = screen.getByText(/prefers espresso over drip coffee/).closest('li')
      expect(li?.className).toContain('border-primary/60')
    })
    const otherLi = screen.getByText(/The Seine flows through Paris/).closest('li')
    expect(otherLi?.className).not.toContain('border-primary/60')
  })

  it('auto-clears the highlight after the timeout', async () => {
    vi.useFakeTimers()
    try {
      render(<MemoryTab />)
      await act(async () => {})
      await act(async () => {})
      publishMemoryEvent({ stored: true, fact: 'The user prefers espresso over drip coffee.' })
      await act(async () => {})
      await act(async () => {})
      const li = screen.getByText(/prefers espresso over drip coffee/).closest('li')
      expect(li?.className).toContain('border-primary/60')
      act(() => { vi.advanceTimersByTime(4000) })
      const liAfter = screen.getByText(/prefers espresso over drip coffee/).closest('li')
      expect(liAfter?.className).not.toContain('border-primary/60')
    } finally {
      vi.useRealTimers()
    }
  })

  it('pre-fills the edit form with the current values', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    expect((screen.getByLabelText('Edit memory fact text') as HTMLTextAreaElement).value).toBe('The user prefers espresso over drip coffee.')
    expect((screen.getByLabelText('Edit memory fact topic') as HTMLInputElement).value).toBe('preferences')
    expect(screen.getByText('3.0')).toBeDefined()
  })

  it('edits a memory item and refreshes the list', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'The user prefers pour-over coffee.' } })
    fireEvent.change(screen.getByLabelText('Edit memory fact topic'), { target: { value: 'coffee' } })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => {
      expect(hoisted.memoryController.update).toHaveBeenCalledWith('m1', 'The user prefers pour-over coffee.', 'coffee', 3)
    })
    await waitFor(() => {
      expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2)
    })
    expect(screen.queryByLabelText('Edit memory fact text')).toBeNull()
  })

  it('shows an inline error when the edit produces a duplicate', async () => {
    hoisted.memoryController.update.mockResolvedValue({ updated: 0, duplicate: true })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'Same as another fact.' } })
    fireEvent.click(screen.getByText('Save'))
    expect(await screen.findByText('That fact already exists in memory')).toBeDefined()
    expect(hoisted.memoryController.list).toHaveBeenCalledTimes(1)
    expect(screen.getByLabelText('Edit memory fact text')).toBeDefined()
  })

  it('cancels editing without saving', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'Changed but cancelled.' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(hoisted.memoryController.update).not.toHaveBeenCalled()
    expect(screen.queryByLabelText('Edit memory fact text')).toBeNull()
  })

  it('reports a failed update inline', async () => {
    hoisted.memoryController.update.mockRejectedValue(new Error('boom'))
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'Updated text.' } })
    fireEvent.click(screen.getByText('Save'))
    expect(await screen.findByText('Could not update memory item')).toBeDefined()
  })

  it('shows search score when results are from search', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', score: 0.42 })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    expect(screen.queryByText('0.42')).toBeNull()
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'espresso' } })
    await screen.findByText('0.42')
    expect(screen.queryByText('0.80')).toBeNull()
  })

  it('hides the score badge once search is cleared', async () => {
    hoisted.memoryController.search.mockResolvedValue({ results: [item({ id: 's1', content: 'Scored hit.', score: 0.77 })], total: 1 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'hit' } })
    await screen.findByText('0.77')
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: '' } })
    await waitFor(() => expect(screen.queryByText('0.77')).toBeNull())
  })

  it('consolidates memory and reports duplicates removed', async () => {
    hoisted.memoryController.consolidate.mockResolvedValue({ removed: 2, kept: 3, threshold: 0.9 })
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('Consolidate'))
    await waitFor(() => expect(hoisted.memoryController.consolidate).toHaveBeenCalledTimes(1))
    expect(await screen.findByText('Consolidated 2 duplicate fact(s), kept 3')).toBeDefined()
    await waitFor(() => expect(hoisted.memoryController.list).toHaveBeenCalledTimes(2))
  })

  it('reports when consolidation finds no near-duplicates', async () => {
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('Consolidate'))
    expect(await screen.findByText('No near-duplicate facts found')).toBeDefined()
  })

  it('reports a consolidate failure inline', async () => {
    hoisted.memoryController.consolidate.mockRejectedValue(new Error('boom'))
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('Consolidate'))
    expect(await screen.findByText('Could not consolidate memory')).toBeDefined()
  })

  it('disables the consolidate button while running', async () => {
    let release!: (v: { removed: number; kept: number; threshold: number }) => void
    hoisted.memoryController.consolidate.mockImplementation(() => new Promise(res => { release = res }))
    render(<MemoryTab />)
    await screen.findByText(/prefers espresso over drip coffee/)
    fireEvent.click(screen.getByText('Consolidate'))
    await waitFor(() => expect((screen.getByText('Consolidating…') as HTMLButtonElement).disabled).toBe(true))
    await act(async () => { release({ removed: 0, kept: 1, threshold: 0.9 }) })
    expect(screen.getByText('Consolidate')).toBeDefined()
  })

  it('hides the consolidate button when no facts exist', async () => {
    hoisted.memoryController.list.mockResolvedValue({ items: [], total: 0 })
    render(<MemoryTab />)
    await screen.findByText('0 facts')
    expect(screen.queryByText('Consolidate')).toBeNull()
  })
})
