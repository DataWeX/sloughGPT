import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockStats = vi.fn()
const mockTopics = vi.fn()
const mockList = vi.fn()
const mockAdd = vi.fn()
const mockSuggestTopic = vi.fn()
const mockSearch = vi.fn()
const mockRemove = vi.fn()
const mockBatchDelete = vi.fn()
const mockUpdate = vi.fn()
const mockGaps = vi.fn()
const mockIngestUrl = vi.fn()

vi.mock('@/lib/kb-controller', () => ({
  kbController: {
    stats: (...args: unknown[]) => mockStats(...args),
    topics: (...args: unknown[]) => mockTopics(...args),
    list: (...args: unknown[]) => mockList(...args),
    add: (...args: unknown[]) => mockAdd(...args),
    suggestTopic: (...args: unknown[]) => mockSuggestTopic(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    remove: (...args: unknown[]) => mockRemove(...args),
    batchDelete: (...args: unknown[]) => mockBatchDelete(...args),
    update: (...args: unknown[]) => mockUpdate(...args),
    gaps: (...args: unknown[]) => mockGaps(...args),
    ingestUrl: (...args: unknown[]) => mockIngestUrl(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant }: any) => <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Input: ({ value, onChange, placeholder, onKeyDown, type }: any) => <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} type={type} />,
    Label: ({ children }: any) => <label>{children}</label>,
    Textarea: ({ value, onChange, placeholder, rows }: any) => <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children }: any) => <span>{children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh" />,
    IconTrash: () => <span data-testid="icon-trash" />,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <p>{children}</p>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p>{headerRight && <div>{headerRight}</div>}{children}</div>
  ),
}))

import KbPage from './page'

describe('KbPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    render(<KbPage />)
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
    expect(screen.getByText('Manage learned facts and knowledge')).toBeInTheDocument()
  })

  it('fetches stats and topics on mount', async () => {
    mockStats.mockResolvedValue({ total_items: 10, topics: ['general', 'code'], avg_importance: 0.7, sources: { manual: 5, url: 5 } })
    mockTopics.mockResolvedValue([{ name: 'general', count: 5 }, { name: 'code', count: 5 }])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
      expect(mockTopics).toHaveBeenCalled()
    })
  })

  it('shows tabs for browse, add, search, gaps', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    expect(screen.getByText('Browse')).toBeInTheDocument()
    expect(screen.getAllByText('Add Entry').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Search').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Knowledge Gaps')).toBeInTheDocument()
  })

  it('browse tab loads items', async () => {
    mockStats.mockResolvedValue({ total_items: 2, topics: ['general'], avg_importance: 0.7, sources: { manual: 2 } })
    mockTopics.mockResolvedValue([{ name: 'general', count: 2 }])
    mockList.mockResolvedValue([
      { id: '1', content: 'Test knowledge', topic: 'general', source: 'manual', importance: 0.8, score: 0 },
      { id: '2', content: 'Another fact', topic: 'general', source: 'manual', importance: 0.6, score: 0 },
    ])
    render(<KbPage />)
    await waitFor(() => {
      expect(screen.getByText('Test knowledge')).toBeInTheDocument()
      expect(screen.getByText('Another fact')).toBeInTheDocument()
    })
  })

  it('add entry', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    mockAdd.mockResolvedValue({ id: '3' })
    render(<KbPage />)

    fireEvent.click(screen.getByText('Browse').parentElement!.children[1])
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Enter knowledge content...')).toBeInTheDocument()
    })

    const textarea = screen.getByPlaceholderText('Enter knowledge content...')
    fireEvent.change(textarea, { target: { value: 'New knowledge' } })
    const addButtons = screen.getAllByText('Add Entry')
    fireEvent.click(addButtons[addButtons.length - 1])

    await waitFor(() => {
      expect(mockAdd).toHaveBeenCalled()
    })
  })

  it('search', async () => {
    mockStats.mockResolvedValue({ total_items: 5, topics: ['general'], avg_importance: 0.7, sources: { manual: 5 } })
    mockTopics.mockResolvedValue([{ name: 'general', count: 5 }])
    mockList.mockResolvedValue([])
    mockSearch.mockResolvedValue([
      { id: '1', content: 'Found result', topic: 'general', source: 'manual', importance: 0.9, score: 0.95 },
    ])
    render(<KbPage />)

    fireEvent.click(screen.getByText('Browse').parentElement!.children[2])
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search knowledge...')).toBeInTheDocument()
    })

    const input = screen.getByPlaceholderText('Search knowledge...')
    fireEvent.change(input, { target: { value: 'test query' } })
    fireEvent.click(screen.getAllByText('Search')[screen.getAllByText('Search').length - 1])

    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('test query', 20)
      expect(screen.getByText('Found result')).toBeInTheDocument()
    })
  })

  it('knowledge gaps', async () => {
    mockStats.mockResolvedValue({ total_items: 5, topics: ['general'], avg_importance: 0.7, sources: { manual: 5 } })
    mockTopics.mockResolvedValue([{ name: 'general', count: 5 }])
    mockList.mockResolvedValue([])
    mockGaps.mockResolvedValue({ gaps: ['No security knowledge', 'No performance data'], suggestions: ['Add security facts', 'Add performance benchmarks'] })
    render(<KbPage />)

    fireEvent.click(screen.getByText('Knowledge Gaps'))
    await waitFor(() => {
      expect(screen.getByText('Analyze Knowledge Gaps')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Analyze Knowledge Gaps'))

    await waitFor(() => {
      expect(mockGaps).toHaveBeenCalled()
      expect(screen.getByText('No security knowledge')).toBeInTheDocument()
      expect(screen.getByText('Add security facts')).toBeInTheDocument()
    })
  })
})