import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockList = vi.fn()
const mockStats = vi.fn()
const mockTopics = vi.fn()
const mockAdd = vi.fn()
const mockRemove = vi.fn()
const mockBatchDelete = vi.fn()
const mockUpdate = vi.fn()
const mockSearch = vi.fn()
const mockSuggestTopic = vi.fn()
const mockGaps = vi.fn()
const mockIngestUrl = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/kb-controller', () => ({
  kbController: {
    list: (...args: unknown[]) => mockList(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    topics: (...args: unknown[]) => mockTopics(...args),
    add: (...args: unknown[]) => mockAdd(...args),
    remove: (...args: unknown[]) => mockRemove(...args),
    batchDelete: (...args: unknown[]) => mockBatchDelete(...args),
    update: (...args: unknown[]) => mockUpdate(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    suggestTopic: (...args: unknown[]) => mockSuggestTopic(...args),
    gaps: (...args: unknown[]) => mockGaps(...args),
    ingestUrl: (...args: unknown[]) => mockIngestUrl(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    label: vi.fn().mockResolvedValue({ label: 'test', confidence: 0.9, reason: '', scores: {} }),
    checkDuplicate: vi.fn().mockResolvedValue({ is_duplicate: false, best_match: null, score: 0, threshold: 0.85 }),
    categorize: vi.fn().mockResolvedValue({ topic: 'test', suggestions: [] }),
    getEmbedderStatus: vi.fn().mockResolvedValue({ trained: false, info: null }),
    trainEmbedder: vi.fn().mockResolvedValue({ status: 'ok', texts_used: 0, epochs: 0, final_loss: 0, save_path: '' }),
    gaps: vi.fn().mockResolvedValue({ gaps: [], total_facts: 0, topics: [] }),
  },
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (_e: unknown, fallback: string) => fallback,
}))

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, confirmLabel }: any) => open ? (
    <div data-testid="alert-dialog"><button data-testid="confirm-action" onClick={onConfirm}>{confirmLabel}</button></div>
  ) : null,
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
    Input: ({ value, onChange, placeholder, className, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} onKeyDown={onKeyDown} />
    ),
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
    Textarea: ({ value, onChange, rows, className, placeholder }: any) => (
      <textarea value={value} onChange={onChange} rows={rows} className={className} placeholder={placeholder} />
    ),
    IconRefresh: (props: any) => <svg {...props} />,
    AlertDialog: ({ children }: any) => <div>{children}</div>,
    AlertDialogTrigger: ({ children }: any) => <div>{children}</div>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    Checkbox: ({ checked, onCheckedChange, className, ...props }: any) => (
      <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} className={className} {...props} />
    ),
    Slider: ({ value, onValueChange, ...props }: any) => (
      <input type="range" value={value?.[0] ?? 0} onChange={(e) => onValueChange?.([Number(e.target.value)])} {...props} />
    ),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, confirmLabel }: any) =>
    open ? <button data-testid="confirm-action" onClick={onConfirm}>{confirmLabel}</button> : null,
}))

import KbPage from './page'

const statsResponse = {
  total_items: 10,
  topics: [{ name: 'general', count: 5 }],
  avg_importance: 0.75,
  sources: { manual: 8, web: 2 },
}

describe('KbPage', () => {
  beforeEach(() => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.restoreAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
    expect(screen.getByText('Manage learned facts and knowledge')).toBeInTheDocument()
  })

  it('fetches stats on mount', async () => {
    mockStats.mockResolvedValue(statsResponse)
    mockTopics.mockResolvedValue([{ name: 'general', count: 5 }])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    await waitFor(() => {
      expect(mockStats).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockTopics).toHaveBeenCalled()
  })

  it('displays stats', async () => {
    mockStats.mockResolvedValue({ total_items: 10, topics: [{ name: 'general', count: 5 }], avg_importance: 0.75, sources: { manual: 10 } })
    mockTopics.mockResolvedValue([{ name: 'general', count: 5 }])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    await waitFor(() => {
      expect(screen.getByText('10')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getAllByText('1')).toHaveLength(2)
    expect(screen.getByText('0.75')).toBeInTheDocument()
  })

  it('renders tab buttons', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    expect(screen.getByText('Browse')).toBeInTheDocument()
    expect(screen.getAllByText('Add Entry').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Search').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Knowledge Gaps')).toBeInTheDocument()
  })

  it('displays knowledge items in browse tab', async () => {
    mockStats.mockResolvedValue({ total_items: 2, topics: [], avg_importance: 0.7, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Fact about AI', topic: 'ai', source: 'manual', importance: 0.9 },
      { id: 'k2', content: 'Fact about coding', topic: 'code', source: 'web', importance: 0.5 },
    ])
    render(<KbPage />)
    await waitFor(() => {
      expect(screen.getByText('Fact about AI')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Fact about coding')).toBeInTheDocument()
  })

  it('shows empty state', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    await waitFor(() => {
      expect(screen.getByText('No entries found.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('adds entry on Add Entry tab', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    mockAdd.mockResolvedValue({})
    render(<KbPage />)
    const addTab = screen.getAllByText('Add Entry')[0]
    fireEvent.click(addTab)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Enter knowledge content...')).toBeInTheDocument()
    })
    fireEvent.change(screen.getByPlaceholderText('Enter knowledge content...'), { target: { value: 'New fact' } })
    const addButtons = screen.getAllByText('Add Entry')
    fireEvent.click(addButtons[addButtons.length - 1])
    await waitFor(() => {
      expect(mockAdd).toHaveBeenCalledWith('New fact', 'general', 'manual', 0.7)
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Entry added', 'success')
  })

  it('searches knowledge', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    mockSearch.mockResolvedValue([
      { id: 'k1', content: 'AI fact', topic: 'ai', source: 'manual', importance: 0.9, score: 0.95 },
    ])
    render(<KbPage />)
    const searchTab = screen.getAllByText('Search')[0]
    fireEvent.click(searchTab)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search knowledge...')).toBeInTheDocument()
    })
    fireEvent.change(screen.getByPlaceholderText('Search knowledge...'), { target: { value: 'AI' } })
    const searchButtons = screen.getAllByText('Search')
    fireEvent.click(searchButtons[searchButtons.length - 1])
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('AI', 20)
    }, { timeout: 5000 })
  })

  it('deletes an entry', async () => {
    mockStats.mockResolvedValue({ total_items: 1, topics: [], avg_importance: 0.7, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Fact', topic: 't', source: 'manual', importance: 0.5 },
    ])
    mockRemove.mockResolvedValue({})
    render(<KbPage />)
    await waitFor(() => {
      expect(screen.getByText('Fact')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => {
      expect(screen.getByTestId('confirm-action')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId('confirm-action'))
    await waitFor(() => {
      expect(mockRemove).toHaveBeenCalledWith('k1')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Entry deleted', 'success')
  })

  it('analyzes knowledge gaps', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    mockGaps.mockResolvedValue({ gaps: ['No coding knowledge'], suggestions: ['Learn Python'] })
    render(<KbPage />)
    fireEvent.click(screen.getByText('Knowledge Gaps'))
    await waitFor(() => {
      expect(screen.getByText('Analyze Knowledge Gaps')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Analyze Knowledge Gaps'))
    await waitFor(() => {
      expect(screen.getByText('No coding knowledge')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Learn Python')).toBeInTheDocument()
  })

  it('shows error on load failure', async () => {
    mockStats.mockRejectedValue(new Error('network'))
    mockTopics.mockRejectedValue(new Error('network'))
    mockList.mockRejectedValue(new Error('network'))
    render(<KbPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Could not load knowledge'),
        'error',
      )
    }, { timeout: 5000 })
  })

  it('refreshes on Refresh click', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topics: [], avg_importance: 0, sources: {} })
    mockTopics.mockResolvedValue([])
    mockList.mockResolvedValue([])
    render(<KbPage />)
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})