import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockList = vi.fn()
const mockStats = vi.fn()
const mockSearch = vi.fn()
const mockStore = vi.fn()
const mockDelete = vi.fn()
const mockUpdate = vi.fn()
const mockConsolidate = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    list: (...args: unknown[]) => mockList(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    store: (...args: unknown[]) => mockStore(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    update: (...args: unknown[]) => mockUpdate(...args),
    consolidate: (...args: unknown[]) => mockConsolidate(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, className, id, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} id={id} onKeyDown={onKeyDown} />
    ),
    Label: ({ children, htmlFor, variant }: any) => <label htmlFor={htmlFor} data-variant={variant}>{children}</label>,
    Progress: ({ value }: any) => <div role="progressbar" aria-valuenow={value} />,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import MemoryPage from './page'

describe('MemoryPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    render(<MemoryPage />)
    expect(screen.getByText('Memory')).toBeInTheDocument()
    expect(screen.getByText('Conversation memory and knowledge retrieval')).toBeInTheDocument()
  })

  it('fetches stats on mount', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 10, topics: 3, visited_urls: 5 })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockList).toHaveBeenCalled()
  })

  it('displays stats', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 10, topics: 3, visited_urls: 5 })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(screen.getByText('10')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
    expect(screen.getByText('Yes')).toBeInTheDocument()
  })

  it('displays memory items', async () => {
    mockList.mockResolvedValue({ items: [{ id: 'm1', content: 'Likes coffee', topic: 'drinks', importance: 0.8 }, { id: 'm2', content: 'Runs daily', topic: 'fitness', importance: 0.5 }] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 2, topics: 2, visited_urls: 0 })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(screen.getByText('Likes coffee')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Runs daily')).toBeInTheDocument()
  })

  it('shows empty state', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(screen.getByText('No memory items.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('searches memory', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    mockSearch.mockResolvedValue({ results: [{ id: 'm1', content: 'Espresso', topic: 'drinks', importance: 0.9 }] })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
    }, { timeout: 5000 })
    fireEvent.change(screen.getByPlaceholderText('Search memory...'), { target: { value: 'espresso' } })
    fireEvent.click(screen.getByText('Search'))
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('espresso', 20)
    }, { timeout: 5000 })
  })

  it('stores a memory', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    mockStore.mockResolvedValue({})
    render(<MemoryPage />)
    fireEvent.click(screen.getByText('Store'))
    fireEvent.change(screen.getByLabelText('Content'), { target: { value: 'New fact' } })
    fireEvent.click(screen.getByText('Store', { selector: 'button' }))
    await waitFor(() => {
      expect(mockStore).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Stored', 'success')
  })

  it('deletes a memory item', async () => {
    mockList.mockResolvedValue({ items: [{ id: 'm1', content: 'Fact', topic: 't', importance: 0.5 }] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 1, topics: 1, visited_urls: 0 })
    mockDelete.mockResolvedValue({})
    render(<MemoryPage />)
    await waitFor(() => {
      expect(screen.getByText('Fact')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Fact'))
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('m1')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Deleted', 'success')
  })

  it('refreshes on Refresh click', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    render(<MemoryPage />)
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })

  it('consolidates memory', async () => {
    mockList.mockResolvedValue({ items: [] })
    mockStats.mockResolvedValue({ enabled: true, total_facts: 0, topics: 0, visited_urls: 0 })
    mockConsolidate.mockResolvedValue({ removed: 3, kept: 7 })
    render(<MemoryPage />)
    fireEvent.click(screen.getByText('Consolidate'))
    await waitFor(() => {
      expect(mockConsolidate).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Consolidated: 3 removed, 7 kept', 'success')
  })

  it('shows error on load failure', async () => {
    mockList.mockRejectedValue(new Error('network'))
    mockStats.mockRejectedValue(new Error('network'))
    render(<MemoryPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load memory', 'error')
    }, { timeout: 5000 })
  })
})
