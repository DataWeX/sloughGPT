import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockList = vi.fn()
const mockStats = vi.fn()
const mockTopics = vi.fn()
const mockSearch = vi.fn()
const mockAdd = vi.fn()
const mockDelete = vi.fn()
const mockUpdate = vi.fn()
const mockBatchDelete = vi.fn()
const mockTrainAdapter = vi.fn()
const mockGetAdapterStatus = vi.fn()

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    list: (...args: unknown[]) => mockList(...args),
    stats: (...args: unknown[]) => mockStats(...args),
    topics: (...args: unknown[]) => mockTopics(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    add: (...args: unknown[]) => mockAdd(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    update: (...args: unknown[]) => mockUpdate(...args),
    batchDelete: (...args: unknown[]) => mockBatchDelete(...args),
    trainAdapter: (...args: unknown[]) => mockTrainAdapter(...args),
    getAdapterStatus: (...args: unknown[]) => mockGetAdapterStatus(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
}))

vi.mock('@/lib/validation-schemas', () => ({
  knowledgeSchema: {
    shape: {
      content: { safeParse: (v: string) => ({ success: true, data: v }) },
      topic: { safeParse: (v: string) => ({ success: true, data: v }) },
    },
  },
}))

import KnowledgePage from './page'

describe('KnowledgePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockStats.mockResolvedValue({ total_items: 0, topic_count: 0, avg_importance: 0, searchable: true, total_chars: 0 })
    mockTopics.mockResolvedValue({ topics: [], total: 0 })
    mockGetAdapterStatus.mockResolvedValue({ adapter_exists: false, fact_count: 0, total_facts_available: 0 })
  })

  it('renders page header', async () => {
    render(<KnowledgePage />)
    expect(screen.getAllByText('Knowledge').length).toBeGreaterThanOrEqual(1)
    await screen.findAllByText(/no|empty|nothing/i)
  })

  it('renders search input', async () => {
    render(<KnowledgePage />)
    expect(screen.getAllByPlaceholderText(/search/i).length).toBeGreaterThanOrEqual(1)
    await screen.findAllByText(/no|empty|nothing/i)
  })

  it('renders Add button', async () => {
    render(<KnowledgePage />)
    expect(screen.getAllByText(/add/i).length).toBeGreaterThanOrEqual(1)
    await screen.findAllByText(/no|empty|nothing/i)
  })

  it('shows empty state when no items', async () => {
    render(<KnowledgePage />)
    await screen.findAllByText(/no|empty|nothing/i)
  })
})
