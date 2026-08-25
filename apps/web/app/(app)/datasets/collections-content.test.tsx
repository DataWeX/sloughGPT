import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: any) => <div data-testid="card">{children}</div>,
  CardContent: ({ children }: any) => <div data-testid="card-content">{children}</div>,
  CardHeader: ({ children }: any) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: any) => <div data-testid="card-title">{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} data-testid="button">{children}</button>
  ),
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label">{children}</label>,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

const { mockApiGet, mockApiPost, mockApiDelete } = vi.hoisted(() => ({
  mockApiGet: vi.fn(),
  mockApiPost: vi.fn(),
  mockApiDelete: vi.fn(),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: any[]) => mockApiGet(...args),
  apiPost: (...args: any[]) => mockApiPost(...args),
  apiDelete: (...args: any[]) => mockApiDelete(...args),
}))

import CollectionsContent from './collections-content'

beforeEach(() => {
  vi.clearAllMocks()
  mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
})

describe('CollectionsContent', () => {
  it('renders without crashing', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/Pipelines/)).toBeDefined()
    })
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<CollectionsContent />)
    const skeletons = screen.getAllByTestId('card')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('shows empty state when no pipelines', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/No pipelines configured/)).toBeDefined()
    })
  })

  it('renders pipeline list from API', async () => {
    mockApiGet.mockResolvedValue({
      pipelines: [
        { id: '1', name: 'test-pipe', source_type: 'file', store_type: 'memory', records_count: 10 },
      ],
      counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 },
    })
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText('test-pipe')).toBeDefined()
    })
    expect(screen.getByText('Source: file')).toBeDefined()
    expect(screen.getByText('Store: memory')).toBeDefined()
  })

  it('renders stats cards when counts are provided', async () => {
    mockApiGet.mockResolvedValue({
      pipelines: [],
      counts: { pipelines: 5, sources: 3, stores: 2, filters: 1 },
    })
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText('5')).toBeDefined()
    })
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })

  it('toggles create form on New pipeline click', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/Pipelines/)).toBeDefined()
    })
    const buttons = screen.getAllByTestId('button')
    const newBtn = buttons.find(b => b.textContent === 'New pipeline')
    expect(newBtn).toBeDefined()
    fireEvent.click(newBtn!)
    expect(screen.getByText('Create pipeline')).toBeDefined()
  })

  it('calls apiGet on mount', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/collections')
    })
  })
})
