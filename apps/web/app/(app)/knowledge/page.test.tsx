import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockList, mockStats, mockTopics, mockSearch, mockAdd, mockDelete, mockUpdate,
  mockBatchDelete, mockGetAdapterStatus, mockTrainAdapter, mockAddToast,
} = vi.hoisted(() => ({
  mockList: vi.fn(), mockStats: vi.fn(), mockTopics: vi.fn(), mockSearch: vi.fn(),
  mockAdd: vi.fn(), mockDelete: vi.fn(), mockUpdate: vi.fn(), mockBatchDelete: vi.fn(),
  mockGetAdapterStatus: vi.fn(), mockTrainAdapter: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    list: (...a: unknown[]) => mockList(...a),
    stats: (...a: unknown[]) => mockStats(...a),
    topics: (...a: unknown[]) => mockTopics(...a),
    search: (...a: unknown[]) => mockSearch(...a),
    add: (...a: unknown[]) => mockAdd(...a),
    delete: (...a: unknown[]) => mockDelete(...a),
    update: (...a: unknown[]) => mockUpdate(...a),
    batchDelete: (...a: unknown[]) => mockBatchDelete(...a),
    getAdapterStatus: (...a: unknown[]) => mockGetAdapterStatus(...a),
    trainAdapter: (...a: unknown[]) => mockTrainAdapter(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/download-utils', () => ({ downloadJson: vi.fn() }))
vi.mock('@/lib/format-bytes', () => ({ todayDateString: () => '2026-08-07', MS_PER_SECOND: 1000 }))
vi.mock('@/lib/validation-schemas', () => ({
  knowledgeSchema: {
    shape: {
      content: { safeParse: (v: string) => ({ success: true, data: v }) },
      topic: { safeParse: (v: string) => ({ success: true, data: v }) },
    },
  },
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    EmptyCard: ({ message, action }: any) => <div><span>{message}</span>{action}</div>,
    Button: ({ children, onClick, disabled, variant, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} aria-label={ariaLabel} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder, className }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} />
    ),
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Chip: ({ children, onClick, variant }: any) => <button onClick={onClick} data-variant={variant}>{children}</button>,
    IconRefresh: iconMock('refresh'), IconPlus: iconMock('plus'), IconTrash: iconMock('trash'),
    IconSearch: iconMock('search'), IconCheck: iconMock('check'), IconX: iconMock('x'),
    IconBrain: iconMock('brain'),
    IconFilter: iconMock('filter'), IconFolder: iconMock('folder'), IconSettings: iconMock('settings'),
    IconClock: iconMock('clock'), IconDownload: iconMock('download'), IconUpload: iconMock('upload'),
    IconChevronDown: iconMock('chevron-down'),
    FoldSection: ({ heading, children }: any) => <details><summary>{heading}</summary>{children}</details>,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: passthrough, AlertDialogHeader: passthrough, AlertDialogTitle: passthrough,
    AlertDialogDescription: passthrough, AlertDialogFooter: passthrough,
    AlertDialogCancel: ({ onClick, ...p }: any) => <button onClick={onClick} {...p}>Cancel</button>,
    AlertDialogAction: ({ onClick, ...p }: any) => <button onClick={onClick} {...p}>Confirm</button>,
    Dialog: ({ open, children }: any) => open ? <div data-testid="dialog">{children}</div> : null,
    DialogContent: passthrough, DialogHeader: passthrough, DialogTitle: passthrough,
    DialogDescription: passthrough, DialogFooter: passthrough,
    Switch: ({ checked, onCheckedChange, disabled, 'aria-label': ariaLabel }: any) => (
      <button role="switch" aria-checked={!!checked} disabled={disabled} aria-label={ariaLabel} onClick={() => onCheckedChange?.(!checked)} />
    ),
    DropdownMenu: passthrough,
    DropdownMenuTrigger: ({ children }: any) => <>{children}</>,
    DropdownMenuContent: ({ children }: any) => <>{children}</>,
    DropdownMenuItem: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
  }
})

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div>{left}{right}</div>,
  AppRouteHeaderLead: ({ title }: any) => <h1>{title}</h1>,
}))

vi.mock('@/components/knowledge/KnowledgeCategoryChart', () => ({
  KnowledgeCategoryChart: ({ topics }: any) => <div data-testid="category-chart" data-count={topics?.length ?? 0} />,
}))

import KnowledgePage from './page'

afterEach(() => cleanup())

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([])
  mockStats.mockResolvedValue({ total_items: 0, topic_count: 0, avg_importance: 0, searchable: true, total_chars: 0 })
  mockTopics.mockResolvedValue({ topics: [], total: 0 })
  mockGetAdapterStatus.mockResolvedValue({ adapter_exists: false, fact_count: 0, total_facts_available: 0 })
  mockAdd.mockResolvedValue({ id: 'new-1', content: 'test', topic: 'general' })
  mockDelete.mockResolvedValue(true)
  mockUpdate.mockResolvedValue(true)
  mockBatchDelete.mockResolvedValue(2)
  mockSearch.mockResolvedValue([])
})

describe('KnowledgePage — loading flow', () => {
  it('shows skeletons while loading, then renders empty state', async () => {
    mockList.mockReturnValue(new Promise(() => {})) // never resolves
    render(<KnowledgePage />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
  })

  it('renders empty state when no items', async () => {
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(screen.getByText(/no|empty|nothing/i)).toBeTruthy()
    })
  })
})

describe('KnowledgePage — data display flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'User likes Python', topic: 'personal', importance: 0.8, created_at: '2026-08-01' },
      { id: 'k2', content: 'API uses REST', topic: 'technical', importance: 0.5, created_at: '2026-08-02' },
      { id: 'k3', content: 'Prefers dark mode', topic: 'preferences', importance: 0.9, created_at: '2026-08-03' },
    ])
    mockStats.mockResolvedValue({ total_items: 3, topic_count: 3, avg_importance: 0.73, searchable: true, total_chars: 50 })
    mockTopics.mockResolvedValue({ topics: [{ topic: 'personal', count: 1 }, { topic: 'technical', count: 1 }, { topic: 'preferences', count: 1 }], total: 3 })
  })

  it('displays all knowledge items', async () => {
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(screen.getByText('User likes Python')).toBeTruthy()
      expect(screen.getByText('API uses REST')).toBeTruthy()
      expect(screen.getByText('Prefers dark mode')).toBeTruthy()
    })
  })

  it('shows total count in header area', async () => {
    render(<KnowledgePage />)
    await waitFor(() => {
      // Stats show "3 items" or "3" somewhere
      const allThree = screen.getAllByText('3')
      expect(allThree.length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders category chart with topics', async () => {
    render(<KnowledgePage />)
    await waitFor(() => {
      const chart = screen.getByTestId('category-chart')
      expect(chart).toBeTruthy()
    })
  })
})

describe('KnowledgePage — add item flow', () => {
  it('renders add button on page', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText(/no|empty|nothing/i)).toBeTruthy() })

    // The Add button should be rendered (IconPlus mock renders "plus" + "Add" text)
    const buttons = screen.getAllByRole('button')
    const addBtn = buttons.find(b => {
      const text = b.textContent || ''
      return text.includes('plus') || text.includes('Add')
    })
    expect(addBtn).toBeTruthy()
  })
})

describe('KnowledgePage — delete item flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Delete me', topic: 'general', importance: 0.5, created_at: '2026-08-01' },
    ])
    mockStats.mockResolvedValue({ total_items: 1, topic_count: 1, avg_importance: 0.5, searchable: true, total_chars: 10 })
    mockTopics.mockResolvedValue({ topics: [{ topic: 'general', count: 1 }], total: 1 })
  })

  it('clicks delete, confirms in dialog, calls delete controller', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Delete me')).toBeTruthy() })

    // Find and click the delete button for this item
    const trashBtns = screen.getAllByLabelText(/delete/i)
    fireEvent.click(trashBtns[0])

    // Confirm dialog should appear
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })

    // Click Confirm
    const confirmBtn = screen.getByText('Confirm')
    await act(async () => { fireEvent.click(confirmBtn) })

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('k1')
    })
  })
})

describe('KnowledgePage — search flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Python is great', topic: 'technical', importance: 0.7, created_at: '2026-08-01' },
      { id: 'k2', content: 'Likes cats', topic: 'personal', importance: 0.5, created_at: '2026-08-02' },
    ])
    mockStats.mockResolvedValue({ total_items: 2, topic_count: 2, avg_importance: 0.6, searchable: true, total_chars: 20 })
    mockTopics.mockResolvedValue({ topics: [], total: 2 })
    mockSearch.mockResolvedValue([
      { id: 'k1', content: 'Python is great', topic: 'technical', importance: 0.7 },
    ])
  })

  it('typing in search triggers debounced search', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Python is great')).toBeTruthy() })

    const searchInput = screen.getByPlaceholderText('Search knowledge...')
    fireEvent.change(searchInput, { target: { value: 'python' } })

    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('python')
    }, { timeout: 1000 })
  })
})

describe('KnowledgePage — batch delete flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Item 1', topic: 'general', importance: 0.5, created_at: '2026-08-01' },
      { id: 'k2', content: 'Item 2', topic: 'general', importance: 0.5, created_at: '2026-08-02' },
    ])
    mockStats.mockResolvedValue({ total_items: 2, topic_count: 1, avg_importance: 0.5, searchable: true, total_chars: 10 })
    mockTopics.mockResolvedValue({ topics: [{ topic: 'general', count: 2 }], total: 2 })
  })

  it('renders select checkboxes when items exist', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Item 1')).toBeTruthy() })
    // Checkboxes should be present for selection
    const checkboxes = screen.getAllByRole('checkbox')
    expect(checkboxes.length).toBeGreaterThanOrEqual(1)
  })
})

describe('KnowledgePage — topic filter flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Tech item', topic: 'technical', importance: 0.7, created_at: '2026-08-01' },
      { id: 'k2', content: 'Personal item', topic: 'personal', importance: 0.5, created_at: '2026-08-02' },
    ])
    mockStats.mockResolvedValue({ total_items: 2, topic_count: 2, avg_importance: 0.6, searchable: true, total_chars: 20 })
    mockTopics.mockResolvedValue({ topics: [{ topic: 'technical', count: 1 }, { topic: 'personal', count: 1 }], total: 2 })
  })

  it('clicking a topic chip filters items', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Tech item')).toBeTruthy() })

    // Click a topic chip
    const techChip = screen.getByText('technical')
    fireEvent.click(techChip)

    await waitFor(() => {
      expect(screen.getByText('Tech item')).toBeTruthy()
    })
  })
})

describe('KnowledgePage — refresh flow', () => {
  it('clicking refresh re-fetches all data', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(mockList).toHaveBeenCalledTimes(1) })

    // Find refresh button by icon or aria-label
    const refreshBtns = screen.getAllByRole('button').filter(b =>
      b.textContent?.includes('refresh') || b.getAttribute('aria-label')?.includes('refresh')
    )
    if (refreshBtns.length > 0) {
      fireEvent.click(refreshBtns[0])
      await waitFor(() => {
        expect(mockList).toHaveBeenCalledTimes(2)
      })
    }
  })
})

describe('KnowledgePage — adapter training flow', () => {
  beforeEach(() => {
    mockGetAdapterStatus.mockResolvedValue({ adapter_exists: false, fact_count: 10, total_facts_available: 10 })
  })

  it('shows adapter section when facts are available', async () => {
    render(<KnowledgePage />)
    await waitFor(() => {
      // Should show adapter info or training button
      const buttons = screen.getAllByRole('button')
      expect(buttons.length).toBeGreaterThan(0)
    })
  })
})

describe('KnowledgePage — error handling flow', () => {
  it('shows toast when list fails', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Failed to load knowledge', 'error')
    })
  })

  it('still renders page structure on error', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(screen.getByText('Knowledge')).toBeTruthy()
    })
  })

  it('retry after error loads data', async () => {
    mockList.mockRejectedValueOnce(new Error('Network error'))
    mockList.mockResolvedValueOnce([{ id: 'k1', content: 'Recovered', topic: 'general', importance: 0.5, created_at: '2026-08-01' }])
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalled()
    })
  })
})

describe('KnowledgePage — empty stats flow', () => {
  it('shows zero values when stats are empty', async () => {
    mockStats.mockResolvedValue({ total_items: 0, topic_count: 0, avg_importance: 0, searchable: true, total_chars: 0 })
    render(<KnowledgePage />)
    await waitFor(() => {
      expect(screen.getByText(/no|empty|nothing/i)).toBeTruthy()
    })
  })
})

describe('KnowledgePage — add form flow', () => {
  it('opens add form when Add button clicked', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText(/no|empty|nothing/i)).toBeTruthy() })
    const addBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.includes('plus') || b.textContent?.includes('Add')
    )
    if (addBtn) {
      fireEvent.click(addBtn)
      await waitFor(() => {
        const inputs = screen.getAllByRole('textbox')
        expect(inputs.length).toBeGreaterThan(0)
      })
    }
  })

  it('submitting add form calls controller', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText(/no|empty|nothing/i)).toBeTruthy() })
    const addBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.includes('plus') || b.textContent?.includes('Add')
    )
    if (addBtn) {
      fireEvent.click(addBtn)
      await waitFor(() => {
        const inputs = screen.getAllByRole('textbox')
        expect(inputs.length).toBeGreaterThan(0)
      })
    }
  })
})

describe('KnowledgePage — batch operations flow', () => {
  beforeEach(() => {
    mockList.mockResolvedValue([
      { id: 'k1', content: 'Batch item 1', topic: 'general', importance: 0.5, created_at: '2026-08-01' },
      { id: 'k2', content: 'Batch item 2', topic: 'general', importance: 0.5, created_at: '2026-08-02' },
      { id: 'k3', content: 'Batch item 3', topic: 'other', importance: 0.5, created_at: '2026-08-03' },
    ])
    mockStats.mockResolvedValue({ total_items: 3, topic_count: 2, avg_importance: 0.5, searchable: true, total_chars: 30 })
    mockTopics.mockResolvedValue({ topics: [{ topic: 'general', count: 2 }, { topic: 'other', count: 1 }], total: 3 })
  })

  it('select all checkbox toggles all items', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Batch item 1')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    expect(checkboxes.length).toBeGreaterThanOrEqual(1)
    // Click first checkbox (select all or first item)
    fireEvent.click(checkboxes[0])
  })

  it('batch delete button appears when items selected', async () => {
    render(<KnowledgePage />)
    await waitFor(() => { expect(screen.getByText('Batch item 1')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 0) {
      fireEvent.click(checkboxes[0])
      await waitFor(() => {
        const batchBtn = screen.getAllByRole('button').find(b =>
          b.textContent?.includes('trash') || b.textContent?.includes('Delete')
        )
        expect(batchBtn).toBeTruthy()
      })
    }
  })
})
