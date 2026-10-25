import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockPush, bootstrapState, mockSessionList, mockModelList, mockApiGet } = vi.hoisted(
  () => ({
    mockPush: vi.fn(),
    bootstrapState: { payload: null as any },
    mockSessionList: vi.fn(),
    mockModelList: vi.fn(),
    mockApiGet: vi.fn(),
  }),
)

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
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
    mockSessionList.mockResolvedValue([])
    mockModelList.mockResolvedValue([])
    mockApiGet.mockResolvedValue({ hits: [], partial: [], skipped: [], query: '' })
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

  it('queries system search with q and renders Data results', async () => {
    mockApiGet.mockResolvedValue({
      hits: [
        {
          id: 'k1',
          store: 'knowledge',
          title: 'Paris is the capital of France',
          detail: 'geography',
          score: 1.0,
          locator: 'route:/knowledge',
        },
      ],
      partial: [],
      skipped: [],
      query: 'paris',
    })
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    fireEvent.change(input, { target: { value: 'paris' } })

    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/search?q=paris&workspace_id=ws-1')
    })
    await waitFor(() => {
      expect(screen.getByText('Paris is the capital of France')).toBeDefined()
    })
    expect(screen.getByText('Data')).toBeDefined()
    // Jumping navigates to the hit's locator route.
    fireEvent.click(screen.getByText('Paris is the capital of France'))
    expect(mockPush).toHaveBeenCalledWith('/knowledge')
  })

  it('shows partial notice when a store fails', async () => {
    mockApiGet.mockResolvedValue({ hits: [], partial: ['knowledge'], skipped: [], query: 'paris' })
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    fireEvent.change(input, { target: { value: 'paris' } })

    await waitFor(() => {
      expect(screen.getByText(/Partial results/)).toBeDefined()
    })
  })

  it('opens file-backed hits in the search results page', async () => {
    mockApiGet.mockResolvedValue({
      hits: [
        {
          id: 'f1',
          store: 'docs',
          title: 'guide.md',
          detail: 'Run the dev stack first.',
          score: 1.0,
          locator: 'file:guide.md:2',
        },
      ],
      partial: [],
      skipped: [],
      query: 'setup',
    })
    render(<CommandPalette />)
    fireEvent.keyDown(window, { key: 'k', metaKey: true })
    const input = screen.getByPlaceholderText('Search pages, models, actions...')
    fireEvent.change(input, { target: { value: 'setup' } })

    await waitFor(() => {
      expect(screen.getByText('guide.md')).toBeDefined()
    })
    fireEvent.click(screen.getByText('guide.md'))
    expect(mockPush).toHaveBeenCalledWith('/workspace/data/search?q=guide.md')
  })
})
