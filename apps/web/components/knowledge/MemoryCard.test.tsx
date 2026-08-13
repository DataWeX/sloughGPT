// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act, within } from '@testing-library/react'
import React from 'react'
import { MemoryCard } from './MemoryCard'
import type { MemoryItem, MemoryStats } from '@/lib/memory-controller'

const { mockList, mockStats, mockSearch, mockStore, mockClear, mockDelete, mockUpdate, mockAddToast, mockArchiveStats, mockConsolidate, mockArchivePrune, mockSetEnabled, mockArchive, mockDownloadJson, mockImportFile, mockGetConfig, mockUpdateConfig } = vi.hoisted(() => ({
  mockList: vi.fn(),
  mockStats: vi.fn(),
  mockSearch: vi.fn(),
  mockStore: vi.fn(),
  mockClear: vi.fn(),
  mockDelete: vi.fn(),
  mockUpdate: vi.fn(),
  mockAddToast: vi.fn(),
  mockArchiveStats: vi.fn(),
  mockConsolidate: vi.fn(),
  mockArchivePrune: vi.fn(),
  mockSetEnabled: vi.fn(),
  mockArchive: vi.fn(),
  mockDownloadJson: vi.fn(),
  mockImportFile: vi.fn(),
  mockGetConfig: vi.fn(),
  mockUpdateConfig: vi.fn(),
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    list: mockList,
    stats: mockStats,
    search: mockSearch,
    store: mockStore,
    remember: vi.fn(),
    clear: mockClear,
    delete: mockDelete,
    update: mockUpdate,
    setEnabled: mockSetEnabled,
    consolidate: mockConsolidate,
    archive: mockArchive,
    archiveStats: mockArchiveStats,
    archivePrune: mockArchivePrune,
    getConfig: mockGetConfig,
    updateConfig: mockUpdateConfig,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: mockDownloadJson,
  importFile: mockImportFile,
}))

const item: MemoryItem = {
  id: 'm1',
  content: 'User prefers espresso in the morning',
  topic: 'preferences',
  source: 'task',
  url: '',
  timestamp: 1700000000,
  importance: 0.8,
  score: 0.9,
}

const geographyItem: MemoryItem = {
  id: 'm2',
  content: 'Stockholm is the capital of Sweden',
  topic: 'geography',
  source: 'task',
  url: '',
  timestamp: 1700000000,
  importance: 0.7,
  score: 0.85,
}

const stats: MemoryStats = {
  enabled: true,
  total_facts: 3,
  topics: 2,
  visited_urls: 4,
}

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue({ items: [item], total: 1 })
  mockStats.mockResolvedValue(stats)
  mockSearch.mockResolvedValue({ results: [], total: 0 })
  mockStore.mockResolvedValue({ stored: true, content: '', topic: 'manual', source: 'api' })
  mockClear.mockResolvedValue({ cleared: 3 })
  mockDelete.mockResolvedValue({ deleted: 1 })
  mockUpdate.mockResolvedValue({ updated: 1, duplicate: false })
  mockArchiveStats.mockResolvedValue({ path: '/tmp/facts.jsonl', records: 5, bytes: 10240, task_types: {}, oldest_ts: 1, newest_ts: 2 })
  mockConsolidate.mockResolvedValue({ removed: 2, kept: 3, threshold: 0.8 })
  mockArchivePrune.mockResolvedValue({ pruned: 4 })
  mockSetEnabled.mockResolvedValue({ enabled: true })
  mockArchive.mockResolvedValue({ records: [], total: 0 })
  mockImportFile.mockResolvedValue(null)
  mockGetConfig.mockResolvedValue({ enabled: true, archive_retention_days: 30, min_chars: 80, max_facts: 5, store_path: 'data/memory', sync_remember: false, consolidation_threshold: 0.8, maintenance_interval_minutes: 60 })
  mockUpdateConfig.mockResolvedValue({ enabled: true, archive_retention_days: 14, min_chars: 80, max_facts: 5, store_path: 'data/memory', sync_remember: false, consolidation_threshold: 0.8, maintenance_interval_minutes: 60 })
})

afterEach(() => {
  cleanup()
})

describe('MemoryCard', () => {
  it('loads stats and items on mount', async () => {
    render(<MemoryCard />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
      expect(mockList).toHaveBeenCalled()
    })
    expect(screen.getAllByText('Memory').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('User prefers espresso in the morning').length).toBeGreaterThanOrEqual(1)
  })

  it('shows stats numbers', async () => {
    render(<MemoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('4').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Off badge when disabled', async () => {
    mockStats.mockResolvedValue({ ...stats, enabled: false })
    render(<MemoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Off').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows empty state when nothing stored', async () => {
    mockList.mockResolvedValue({ items: [], total: 0 })
    mockStats.mockResolvedValue({ ...stats, total_facts: 0 })
    render(<MemoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/Nothing remembered yet/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('searches memory after typing with debounce', async () => {
    mockSearch.mockResolvedValue({ results: [item], total: 1 })
    render(<MemoryCard />)
    const input = await screen.findByPlaceholderText('Search memory...')
    fireEvent.change(input, { target: { value: 'espresso' } })
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('espresso')
    }, { timeout: 2000 })
  })

  it('shows search score when results are from search', async () => {
    mockSearch.mockResolvedValue({ results: [item], total: 1 })
    render(<MemoryCard />)
    const input = await screen.findByPlaceholderText('Search memory...')
    fireEvent.change(input, { target: { value: 'espresso' } })
    await waitFor(() => {
      expect(screen.getAllByText('0.90').length).toBeGreaterThanOrEqual(1)
    }, { timeout: 2000 })
  })

  it('renders topic chips derived from the loaded items', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    expect(screen.getAllByText('All')[0]).toBeTruthy()
    expect(screen.getAllByText('geography').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('preferences').length).toBeGreaterThanOrEqual(1)
  })

  it('filters the list when a topic chip is selected', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByText('geography')[0])
    expect(screen.queryByText('User prefers espresso in the morning')).toBeNull()
    expect(screen.getAllByText('Stockholm is the capital of Sweden').length).toBeGreaterThanOrEqual(1)
  })

  it('shows the empty message when the active topic has no items', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    mockSearch.mockResolvedValue({ results: [], total: 0 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByText('geography')[0])
    const input = screen.getByPlaceholderText('Search memory...')
    fireEvent.change(input, { target: { value: 'zzz-no-match' } })
    await waitFor(() => {
      expect(screen.getByText('No memory in the "geography" topic.')).toBeTruthy()
    }, { timeout: 2000 })
  })

  it('resets the filter when clicking All', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByText('geography')[0])
    expect(screen.queryByText('User prefers espresso in the morning')).toBeNull()

    fireEvent.click(screen.getAllByText('All')[0])
    expect(screen.getAllByText('User prefers espresso in the morning').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Stockholm is the capital of Sweden').length).toBeGreaterThanOrEqual(1)
  })

  it('expands the list past 10 items with Show all', async () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ ...item, id: `m${i}`, content: `Fact number ${i + 1}`, topic: 'bulk' }))
    mockList.mockResolvedValue({ items: many, total: 12 })
    render(<MemoryCard />)
    await screen.findByText('Fact number 1')

    expect(screen.queryByText('Fact number 12')).toBeNull()
    expect(screen.getAllByText(/Showing 10 of 12/).length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getAllByText('Show all')[0])
    expect(screen.getAllByText('Fact number 12').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Show fewer').length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getAllByText('Show fewer')[0])
    expect(screen.queryByText('Fact number 12')).toBeNull()
  })

  it('offers a Clear search recovery action when nothing matches', async () => {
    mockSearch.mockResolvedValue({ results: [], total: 0 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    const input = screen.getByPlaceholderText('Search memory...')
    fireEvent.change(input, { target: { value: 'nope' } })
    await screen.findByText('No memory matches that search.')

    fireEvent.click(screen.getAllByText('Clear search')[0])
    expect(screen.getByPlaceholderText('Search memory...')).toHaveValue('')
    expect(screen.getAllByText('User prefers espresso in the morning').length).toBeGreaterThanOrEqual(1)
  })

  it('shows relative timestamps on items', async () => {
    const recent = { ...item, id: 'm3', content: 'A very recent fact', timestamp: Math.floor(Date.now() / 1000) - 60 }
    mockList.mockResolvedValue({ items: [recent], total: 1 })
    render(<MemoryCard />)
    await screen.findByText('A very recent fact')
    expect(screen.getAllByText('1m ago').length).toBeGreaterThanOrEqual(1)
  })

  it('sorts facts newest first by default and flips to oldest', async () => {
    const old = { ...item, id: 'm-old', content: 'Old fact', timestamp: 1600000000 }
    const mid = { ...item, id: 'm-mid', content: 'Mid fact', timestamp: 1650000000 }
    const recent = { ...item, id: 'm-new', content: 'Recent fact', timestamp: 1700000000 }
    mockList.mockResolvedValue({ items: [old, mid, recent], total: 3 })
    render(<MemoryCard />)
    await screen.findByText('Recent fact')

    const order = () => screen.getAllByTitle('Click to copy').map(el => el.textContent)
    expect(order()).toEqual(['Recent fact', 'Mid fact', 'Old fact'])

    fireEvent.click(screen.getByLabelText('Toggle memory sort order'))
    const oldest = await screen.findByRole('menuitem', { name: 'Oldest' })
    fireEvent.click(oldest)
    await waitFor(() => expect(screen.getAllByText('Oldest').length).toBeGreaterThanOrEqual(1))
    expect(order()).toEqual(['Old fact', 'Mid fact', 'Recent fact'])
  })

  it('sorts facts by importance when selected from the sort menu', async () => {
    const low = { ...item, id: 'm-low', content: 'Low importance fact', timestamp: 1600000000, importance: 0.2 }
    const high = { ...item, id: 'm-high', content: 'High importance fact', timestamp: 1700000000, importance: 0.9 }
    const mid = { ...item, id: 'm-mid', content: 'Mid importance fact', timestamp: 1650000000, importance: 0.5 }
    mockList.mockResolvedValue({ items: [low, high, mid], total: 3 })
    render(<MemoryCard />)
    await screen.findByText('High importance fact')

    fireEvent.click(screen.getByLabelText('Toggle memory sort order'))
    const importance = await screen.findByRole('menuitem', { name: 'Importance' })
    fireEvent.click(importance)
    await waitFor(() => expect(screen.getAllByText('Importance').length).toBeGreaterThanOrEqual(1))

    const order = () => screen.getAllByTitle('Click to copy').map(el => el.textContent)
    expect(order()).toEqual(['High importance fact', 'Mid importance fact', 'Low importance fact'])
  })

  it('copies a fact to the clipboard when clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    mockList.mockResolvedValue({ items: [item], total: 1 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByText('User prefers espresso in the morning'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('User prefers espresso in the morning'))
    expect(mockAddToast).toHaveBeenCalledWith('Memory fact copied', 'success')
  })

  it('edits a memory fact and saves via the API', async () => {
    mockList.mockResolvedValue({ items: [item], total: 1 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByLabelText('Edit memory item')[0])
    expect(screen.getByLabelText('Edit memory fact text')).toHaveValue('User prefers espresso in the morning')

    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'User prefers a cappuccino after noon' } })
    fireEvent.change(screen.getByPlaceholderText('preferences'), { target: { value: 'drinks' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => expect(mockUpdate).toHaveBeenCalledWith('m1', 'User prefers a cappuccino after noon', 'drinks', 0.8))
    expect(mockAddToast).toHaveBeenCalledWith('Memory item updated', 'success')
    expect(screen.queryByLabelText('Edit memory fact text')).toBeNull()
  })

  it('edits a memory fact importance via the slider and saves via the API', async () => {
    mockList.mockResolvedValue({ items: [item], total: 1 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    expect(screen.getAllByText('importance 0.8').length).toBeGreaterThanOrEqual(1)

    fireEvent.click(screen.getAllByLabelText('Edit memory item')[0])
    const slider = screen.getByLabelText('Edit memory fact importance') as HTMLInputElement
    expect(slider.value).toBe('0.8')

    fireEvent.change(slider, { target: { value: '0.3' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => expect(mockUpdate).toHaveBeenCalledWith('m1', 'User prefers espresso in the morning', 'preferences', 0.3))
    expect(mockAddToast).toHaveBeenCalledWith('Memory item updated', 'success')
  })

  it('surfaces a duplicate error when the edited fact already exists', async () => {
    mockUpdate.mockResolvedValue({ updated: 0, duplicate: true })
    mockList.mockResolvedValue({ items: [item], total: 1 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByLabelText('Edit memory item')[0])
    fireEvent.change(screen.getByLabelText('Edit memory fact text'), { target: { value: 'Duplicated fact text' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => expect(mockAddToast).toHaveBeenCalledWith('That fact already exists in memory', 'error'))
  })

  it('stores a fact via the inline form', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByText('Store fact')[0])
    const textarea = screen.getAllByLabelText('New memory fact')[0]
    fireEvent.change(textarea, { target: { value: 'User codes in TypeScript' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => {
      expect(mockStore).toHaveBeenCalledWith('User codes in TypeScript', 'manual')
    })
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Saved to memory', 'success')
  })

  it('disables Save when empty', async () => {
    render(<MemoryCard />)
    fireEvent.click(screen.getAllByText('Store fact')[0])
    const save = screen.getAllByText('Save')[0].closest('button') as HTMLButtonElement
    expect(save.disabled).toBe(true)
  })

  it('clears all memory after confirmation', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByText('Clear')[0])
    const confirm = await screen.findAllByText('Clear all')
    await act(async () => {
      fireEvent.click(confirm[confirm.length - 1])
    })

    await waitFor(() => {
      expect(mockClear).toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Cleared 3 memory items', 'success')
  })

  it('deletes a single memory item after confirmation', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getAllByLabelText('Delete memory item')[0])
    const confirm = await screen.findAllByText('Delete')
    await act(async () => {
      fireEvent.click(confirm[confirm.length - 1])
    })

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('m1')
    })
    expect(mockAddToast).toHaveBeenCalledWith('Memory item deleted', 'success')
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('selects memory facts and deletes them in batch after confirmation', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await screen.findByText('Stockholm is the capital of Sweden')

    fireEvent.click(screen.getByLabelText('Select memory fact User prefers espresso in the morning'))
    fireEvent.click(screen.getByLabelText('Select memory fact Stockholm is the capital of Sweden'))

    await act(async () => {
      fireEvent.click(screen.getByText('Delete (2)'))
    })
    const confirm = await screen.findAllByText('Delete')
    await act(async () => {
      fireEvent.click(confirm[confirm.length - 1])
    })

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledTimes(2)
    })
    expect(mockDelete).toHaveBeenCalledWith('m1')
    expect(mockDelete).toHaveBeenCalledWith('m2')
    expect(mockAddToast).toHaveBeenCalledWith('Deleted 2 memory item(s)', 'success')
    expect(screen.queryByText('Delete (2)')).toBeNull()
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('reports a partial batch delete when selected items are missing', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    mockDelete.mockResolvedValue({ deleted: 0 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByLabelText('Select memory fact User prefers espresso in the morning'))
    await act(async () => {
      fireEvent.click(screen.getByText('Delete (1)'))
    })
    const confirm = await screen.findAllByText('Delete')
    await act(async () => {
      fireEvent.click(confirm[confirm.length - 1])
    })

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('m1')
    })
    expect(mockAddToast).toHaveBeenCalledWith('Selected items not found', 'error')
  })

  it('selects all facts with the select-all checkbox', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    const selectAll = screen.getByLabelText('Select all memory facts')
    expect(selectAll).not.toBeChecked()
    fireEvent.click(selectAll)
    expect(selectAll).toBeChecked()
    expect(screen.getByLabelText('Select memory fact User prefers espresso in the morning')).toBeChecked()
    expect(screen.getByLabelText('Select memory fact Stockholm is the capital of Sweden')).toBeChecked()

    fireEvent.click(selectAll)
    expect(selectAll).not.toBeChecked()
    expect(screen.getByLabelText('Select memory fact User prefers espresso in the morning')).not.toBeChecked()
  })

  it('exports only the selected facts as JSON', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByLabelText('Select memory fact Stockholm is the capital of Sweden'))
    fireEvent.click(screen.getByText('Export (1)'))

    expect(mockDownloadJson).toHaveBeenCalledWith(
      [{ content: geographyItem.content, topic: 'geography', source: 'task' }],
      expect.stringContaining('memory-export-selected-'),
    )
    expect(mockAddToast).toHaveBeenCalledWith('Exported 1 memory item(s)', 'success')
  })

  it('clears the selection when Cancel is clicked', async () => {
    mockList.mockResolvedValue({ items: [item, geographyItem], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByLabelText('Select memory fact User prefers espresso in the morning'))
    expect(screen.getByText('Delete (1)')).toBeTruthy()
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Delete (1)')).toBeNull()
    expect(screen.getByLabelText('Select memory fact User prefers espresso in the morning')).not.toBeChecked()
  })

  const openMaintenance = async () => {
    fireEvent.click(screen.getAllByText('Maintenance')[0])
    await waitFor(() => {
      expect(screen.getAllByText('Consolidate duplicates').length).toBeGreaterThanOrEqual(1)
    })
  }

  it('loads archive stats on mount', async () => {
    render(<MemoryCard />)
    await waitFor(() => {
      expect(mockArchiveStats).toHaveBeenCalled()
    })
  })

  it('consolidates duplicates and reports the result', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Consolidate')[0])

    await waitFor(() => {
      expect(mockConsolidate).toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Consolidated 2 duplicate fact(s), kept 3', 'success')
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('reports when no near-duplicates are found', async () => {
    mockConsolidate.mockResolvedValue({ removed: 0, kept: 1, threshold: 0.8 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Consolidate')[0])

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('No near-duplicate facts found', 'info')
    })
  })

  it('shows archive record count', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    await waitFor(() => {
      expect(screen.getAllByText(/record\(s\)/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('prunes the archive and reports the result', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Prune old')[0])

    await waitFor(() => {
      expect(mockArchivePrune).toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Pruned 4 archive record(s)', 'success')
  })

  it('loads the configured retention into the maintenance input', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()

    const input = screen.getByLabelText('Archive retention days')
    expect(input).toHaveValue(30)
    expect(screen.getAllByText('Archive retention').length).toBeGreaterThanOrEqual(1)
  })

  it('saves a new retention window via updateConfig', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()

    const input = screen.getByLabelText('Archive retention days')
    fireEvent.change(input, { target: { value: '14' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => {
      expect(mockUpdateConfig).toHaveBeenCalledWith({ archive_retention_days: 14 })
    })
    expect(mockAddToast).toHaveBeenCalledWith('Archive retention set to 14 day(s)', 'success')
  })

  it('prunes using the configured retention window', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    await screen.findByLabelText('Archive retention days')

    fireEvent.click(screen.getAllByText('Prune old')[0])
    await waitFor(() => {
      expect(mockArchivePrune).toHaveBeenCalledWith(30)
    })
  })

  it('rejects saving an empty retention input', async () => {
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()

    const input = screen.getByLabelText('Archive retention days')
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.click(screen.getAllByText('Save')[0])

    await waitFor(() => {
      expect(mockUpdateConfig).not.toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Enter a retention window in days', 'error')
  })

  it('opens the archive browser and lists records', async () => {
    mockArchive.mockResolvedValue({
      records: [
        { task_type: 'memory.remember', task_id: 'r1', ts: 1700000000, user_message: 'User prefers espresso in the morning' },
        { task_type: 'memory.store', task_id: 's1', ts: 1700000000, content: 'Stockholm is the capital of Sweden', topic: 'geography' },
        { task_type: 'memory.consolidate', task_id: 'c1', ts: 1700000000, removed: 2, kept: 3, threshold: 0.8 },
        { task_type: 'memory.forget', task_id: 'f1', ts: 1700000000 },
      ],
      total: 4,
    })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('View records')[0])

    await waitFor(() => {
      expect(mockArchive).toHaveBeenCalledWith(20)
    })
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('Provenance archive')).toBeTruthy()
    expect(within(dialog).getAllByText(/Stockholm is the capital of Sweden/)).toBeTruthy()
    expect(within(dialog).getByText(/Consolidated 2 duplicate\(s\), kept 3/)).toBeTruthy()
    expect(within(dialog).getByText('Topic: geography')).toBeTruthy()
    expect(within(dialog).getByText('forget')).toBeTruthy()
    expect(within(dialog).getByText('remember')).toBeTruthy()
  })

  it('shows an empty state when the archive has no records', async () => {
    mockArchive.mockResolvedValue({ records: [], total: 0 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('View records')[0])

    expect(await screen.findByText('No archive records yet')).toBeTruthy()
  })

  it('refreshes the archive when clicking Refresh', async () => {
    mockArchive.mockResolvedValue({
      records: [{ task_type: 'memory.store', task_id: 's1', ts: 1700000000, content: 'First pass' }],
      total: 1,
    })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('View records')[0])
    await screen.findByText('First pass')

    mockArchive.mockResolvedValue({
      records: [{ task_type: 'memory.store', task_id: 's2', ts: 1700000000, content: 'After refresh' }],
      total: 1,
    })
    fireEvent.click(within(screen.getByRole('dialog')).getByText('Refresh'))

    await waitFor(() => {
      expect(mockArchive).toHaveBeenCalledTimes(2)
    })
    expect(await screen.findByText('After refresh')).toBeTruthy()
  })

  it('expands an archive record to show its raw payload', async () => {
    mockArchive.mockResolvedValue({
      records: [{ task_type: 'memory.store', task_id: 's1', ts: 1700000000, content: 'Stockholm is the capital of Sweden', topic: 'geography', url: 'https://example.com' }],
      total: 1,
    })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('View records')[0])
    const dialog = screen.getByRole('dialog')
    await within(dialog).findByText('Stockholm is the capital of Sweden')
    fireEvent.click(within(dialog).getByRole('button', { name: /Stockholm is the capital of Sweden/ }))

    expect(await within(dialog).findByText(/"task_id": "s1"/)).toBeTruthy()
    expect(within(dialog).getByText(/"url": "https:\/\/example.com"/)).toBeTruthy()
    fireEvent.click(within(dialog).getByRole('button', { name: /Stockholm is the capital of Sweden/ }))
    await waitFor(() => {
      expect(within(dialog).queryByText(/"task_id": "s1"/)).toBeNull()
    })
  })

  it('exports the archive as JSON from the dialog', async () => {
    mockArchive.mockResolvedValue({
      records: [{ task_type: 'memory.store', task_id: 's1', ts: 1700000000, content: 'Archive fact one' }],
      total: 1,
    })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('View records')[0])
    await screen.findByText('Archive fact one')
    fireEvent.click(within(screen.getByRole('dialog')).getByText('Export'))

    await waitFor(() => {
      expect(mockArchive).toHaveBeenCalledWith(1000)
    })
    expect(mockDownloadJson).toHaveBeenCalledWith(
      [{ task_type: 'memory.store', task_id: 's1', ts: 1700000000, content: 'Archive fact one' }],
      expect.stringContaining('memory-archive-'),
    )
    expect(mockAddToast).toHaveBeenCalledWith('Exported 1 archive record(s)', 'success')
  })

  it('exports all memory as JSON', async () => {
    mockList.mockResolvedValue({ items: [item, { ...item, id: 'm2', content: 'User codes in TypeScript', topic: 'work', source: 'chat' }], total: 2 })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Export')[0])

    await waitFor(() => {
      expect(mockList).toHaveBeenCalledWith(1000)
    })
    expect(mockDownloadJson).toHaveBeenCalledWith(
      [
        { content: item.content, topic: 'preferences', source: 'task' },
        { content: 'User codes in TypeScript', topic: 'work', source: 'chat' },
      ],
      expect.stringContaining('memory-export-'),
    )
    expect(mockAddToast).toHaveBeenCalledWith('Exported 2 memory item(s)', 'success')
  })

  it('imports memory from a JSON file', async () => {
    mockImportFile.mockResolvedValue(new File(['[{"content":"Fact one","topic":"notes"},{"content":"Fact two"}]'], 'memory.json', { type: 'application/json' }))
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Import')[0])

    await waitFor(() => {
      expect(mockStore).toHaveBeenCalledTimes(2)
    })
    expect(mockStore).toHaveBeenCalledWith('Fact one', 'notes')
    expect(mockStore).toHaveBeenCalledWith('Fact two', 'manual')
    expect(mockAddToast).toHaveBeenCalledWith('Imported 2 of 2 memory item(s)', 'success')
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('imports memory from a CSV file with header', async () => {
    mockImportFile.mockResolvedValue(new File(['content,topic\n"Fact from csv","travel"\nSecond fact,'], 'memory.csv', { type: 'text/csv' }))
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Import')[0])

    await waitFor(() => {
      expect(mockStore).toHaveBeenCalledWith('Fact from csv', 'travel')
    })
    expect(mockStore).toHaveBeenCalledWith('Second fact', 'manual')
  })

  it('reports a toast when the import file has no items', async () => {
    mockImportFile.mockResolvedValue(new File(['[]'], 'memory.json', { type: 'application/json' }))
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')
    await openMaintenance()
    fireEvent.click(screen.getAllByText('Import')[0])

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('No memory items found in file', 'error')
    })
    expect(mockStore).not.toHaveBeenCalled()
  })

  it('renders the memory toggle checked when enabled', async () => {
    render(<MemoryCard />)
    await waitFor(() => {
      const toggle = screen.getByRole('switch', { name: 'Toggle automatic memory' })
      expect(toggle.getAttribute('aria-checked')).toBe('true')
    })
  })

  it('reflects the off state in the toggle when disabled', async () => {
    mockStats.mockResolvedValue({ ...stats, enabled: false })
    render(<MemoryCard />)
    await waitFor(() => {
      const toggle = screen.getByRole('switch', { name: 'Toggle automatic memory' })
      expect(toggle.getAttribute('aria-checked')).toBe('false')
    })
  })

  it('disables memory when toggled off', async () => {
    mockSetEnabled.mockResolvedValue({ enabled: false })
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByRole('switch', { name: 'Toggle automatic memory' }))

    await waitFor(() => {
      expect(mockSetEnabled).toHaveBeenCalledWith(false)
    })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Memory is off — the AI will stop storing new facts', 'success')
    })
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('enables memory when toggled on', async () => {
    mockStats.mockResolvedValue({ ...stats, enabled: false })
    mockSetEnabled.mockResolvedValue({ enabled: true })
    render(<MemoryCard />)
    await waitFor(() => {
      const toggle = screen.getByRole('switch', { name: 'Toggle automatic memory' })
      expect(toggle.getAttribute('aria-checked')).toBe('false')
    })

    fireEvent.click(screen.getByRole('switch', { name: 'Toggle automatic memory' }))

    await waitFor(() => {
      expect(mockSetEnabled).toHaveBeenCalledWith(true)
    })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Memory is on — the AI will keep learning from conversations', 'success')
    })
  })

  it('shows an error toast when the toggle fails', async () => {
    mockSetEnabled.mockRejectedValue(new Error('boom'))
    render(<MemoryCard />)
    await screen.findByText('User prefers espresso in the morning')

    fireEvent.click(screen.getByRole('switch', { name: 'Toggle automatic memory' }))

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Failed to update memory setting', 'error')
    })
  })
})
