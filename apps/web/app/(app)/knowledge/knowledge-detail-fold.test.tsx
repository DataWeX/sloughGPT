import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

// ── strui mock ──
vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, disabled, 'aria-label': ariaLabel, className }: any) => (
      <button onClick={onClick} disabled={disabled} aria-label={ariaLabel} className={className}>
        {children}
      </button>
    ),
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children, className }: any) => <span className={className}>{children}</span>,
    IconRefresh: iconMock('refresh'),
  }
})

// ── controller & router mocks ──
const {
  mockApiGet,
  mockKnowledgeList,
  mockKnowledgeUpdate,
  mockKnowledgeDelete,
  mockPush,
  mockAddToast,
} = vi.hoisted(() => ({
  mockApiGet: vi.fn(),
  mockKnowledgeList: vi.fn(),
  mockKnowledgeUpdate: vi.fn(),
  mockKnowledgeDelete: vi.fn(),
  mockPush: vi.fn(),
  mockAddToast: vi.fn(),
}))

const stableRouter = { push: mockPush }
vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => stableRouter,
}))
vi.mock('@/lib/http-client', () => ({ apiGet: mockApiGet }))
vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    list: mockKnowledgeList,
    update: mockKnowledgeUpdate,
    delete: mockKnowledgeDelete,
  },
}))
vi.mock('@/lib/toast-store', () => {
  const mod = {
    useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
  }
  const sink = mod.useToastStore as unknown as ((f: (x: unknown) => unknown) => unknown) & {
    getState?: () => unknown
  }
  if (typeof sink.getState !== 'function') {
    sink.getState = () => {
      const st = sink((x) => x)
      return typeof st === 'object' && st !== null ? st : mod
    }
  }
  return mod
})
vi.mock('@/lib/conversations-utils', () => ({ formatDate: vi.fn(() => 'Jan 1, 2026') }))
import { KnowledgeDetailFold } from './knowledge-detail-fold'

const MOCK_ITEM: any = {
  id: 'kb-1',
  content: 'The user prefers dark mode and likes TypeScript.',
  topic: 'personal',
  source: 'conversation',
  url: 'https://example.com/article/1234567890',
  timestamp: 1704067200000,
  importance: 0.85,
  score: 3.42,
}

afterEach(() => {
  cleanup()
})
beforeEach(() => {
  vi.clearAllMocks()
  mockApiGet.mockResolvedValue(MOCK_ITEM)
})

describe('KnowledgeDetailFold', () => {
  it('shows loading state and fetches item on mount', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<KnowledgeDetailFold id="kb-1" />)
    expect(screen.getByTestId('knowledge-detail-loading')).toBeTruthy()
    expect(mockApiGet).toHaveBeenCalledWith('/knowledge/kb-1')
  })

  it('displays knowledge item after loading', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('The user prefers dark mode and likes TypeScript.')).toBeTruthy()
    })
    expect(screen.getByText('personal')).toBeTruthy()
    expect(screen.getByText('85%')).toBeTruthy()
    expect(screen.getByText('3.42')).toBeTruthy()
    expect(screen.getByText('Jan 1, 2026')).toBeTruthy()
    expect(screen.getByText('conversation')).toBeTruthy()
    expect(screen.getByText('kb-1')).toBeTruthy()
  })

  it('renders page title after data loads', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Knowledge Item')).toBeTruthy()
    })
  })

  it('falls back to knowledgeController.list when apiGet fails', async () => {
    mockApiGet.mockRejectedValue(new Error('not found'))
    mockKnowledgeList.mockResolvedValue([MOCK_ITEM, { ...MOCK_ITEM, id: 'kb-other' }])
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('The user prefers dark mode and likes TypeScript.')).toBeTruthy()
    })
    expect(mockKnowledgeList).toHaveBeenCalled()
  })

  it('shows toast when item not found in list fallback', async () => {
    mockApiGet.mockRejectedValue(new Error('not found'))
    mockKnowledgeList.mockResolvedValue([])
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Knowledge item not found', 'error')
    })
  })

  it('shows error toast when fetch entirely fails', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    mockKnowledgeList.mockRejectedValue(new Error('network'))
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Failed to load knowledge item', 'error')
    })
  })

  it('navigates back via Back button', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Back')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Back').click()
    })
    expect(mockPush).toHaveBeenCalledWith('/knowledge')
  })

  it('deletes item and navigates to knowledge list', async () => {
    mockKnowledgeDelete.mockResolvedValue({})
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Delete').click()
    })
    await waitFor(() => {
      expect(mockKnowledgeDelete).toHaveBeenCalledWith('kb-1')
    })
    expect(mockAddToast).toHaveBeenCalledWith('Deleted', 'success')
    expect(mockPush).toHaveBeenCalledWith('/knowledge')
  })

  it('shows toast on delete failure', async () => {
    mockKnowledgeDelete.mockRejectedValue(new Error('fail'))
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Delete').click()
    })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Delete failed', 'error')
    })
  })

  it('shows url when item has url', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText(/example\.com/)).toBeTruthy()
    })
  })

  it('shows source as manual when no source', async () => {
    mockApiGet.mockResolvedValue({ ...MOCK_ITEM, source: '' })
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('manual')).toBeTruthy()
    })
  })

  it('navigates to chat with context on Use in Chat click', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Use in Chat')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Use in Chat').click()
    })
    expect(mockPush).toHaveBeenCalledWith('/chat?context=kb-1')
  })

  it('opens edit mode when Edit clicked', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Edit')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Edit').click()
    })
    expect(screen.getByText('Save')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
    expect(
      screen.getByDisplayValue('The user prefers dark mode and likes TypeScript.'),
    ).toBeTruthy()
  })

  it('saves edits and exits edit mode', async () => {
    mockKnowledgeUpdate.mockResolvedValue({})
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Edit')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Edit').click()
    })
    await act(async () => {
      screen
        .getByDisplayValue('The user prefers dark mode and likes TypeScript.')
        .setAttribute('value', 'Updated content')
    })
    await act(async () => {
      screen.getByText('Save').click()
    })
    await waitFor(() => {
      expect(mockKnowledgeUpdate).toHaveBeenCalled()
    })
  })

  it('cancels editing without saving', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Edit')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Edit').click()
    })
    await act(async () => {
      screen.getByText('Cancel').click()
    })
    expect(mockKnowledgeUpdate).not.toHaveBeenCalled()
    expect(screen.getByText('Edit')).toBeTruthy()
  })

  it('shows saving state during save', async () => {
    mockKnowledgeUpdate.mockReturnValue(new Promise(() => {}))
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Edit')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Edit').click()
    })
    await act(async () => {
      screen.getByText('Save').click()
    })
    await waitFor(() => {
      const saveBtn = screen.getByText('...')
      expect(saveBtn).toBeTruthy()
    })
  })

  it('renders back button in headerRight', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByTestId('fold-actions')).toBeTruthy()
    })
  })

  it('refetches on refresh button click', async () => {
    render(<KnowledgeDetailFold id="kb-1" />)
    await waitFor(() => {
      expect(screen.getByText('Back')).toBeTruthy()
    })
    await act(async () => {
      const foldActions = screen.getByTestId('fold-actions')
      const refreshBtn = foldActions.querySelector('button:last-child') as HTMLElement
      refreshBtn.click()
    })
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(2)
    })
  })
})
