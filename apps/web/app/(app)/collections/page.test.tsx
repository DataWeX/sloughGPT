import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiDelete = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
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
    Input: ({ value, onChange, placeholder, className, id }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} id={id} />
    ),
    Label: ({ children, htmlFor, variant }: any) => <label htmlFor={htmlFor} data-variant={variant}>{children}</label>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import CollectionsPage from './page'

describe('CollectionsPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
    render(<CollectionsPage />)
    expect(screen.getByText('Collections')).toBeInTheDocument()
    expect(screen.getByText('Data collection pipelines')).toBeInTheDocument()
  })

  it('fetches pipelines on mount', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [{ id: 'p1', name: 'web-scraper', source_type: 'url', store_type: 'memory', records_count: 10 }], counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 } })
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/collections')
    }, { timeout: 5000 })
  })

  it('displays pipeline list', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [{ id: 'p1', name: 'web-scraper', source_type: 'url', store_type: 'memory', records_count: 10 }], counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 } })
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(screen.getByText('web-scraper')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('Source: url')).toBeInTheDocument()
    expect(screen.getByText('Store: memory')).toBeInTheDocument()
  })

  it('displays stats cards', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: { pipelines: 5, sources: 3, stores: 2, filters: 1 } })
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
  })

  it('shows empty state', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(screen.getByText('No pipelines configured. Create one above.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('shows create form', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
    render(<CollectionsPage />)
    fireEvent.click(screen.getByText('New pipeline'))
    expect(screen.getByText('Create pipeline')).toBeInTheDocument()
    expect(screen.getByLabelText('Name')).toBeInTheDocument()
  })

  it('create button disabled when name is empty', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
    render(<CollectionsPage />)
    fireEvent.click(screen.getByText('New pipeline'))
    const btn = screen.getByText('Create', { selector: 'button' })
    expect(btn).toBeDisabled()
  })

  it('runs a pipeline', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [{ id: 'web-scraper', name: 'web-scraper', source_type: 'url', store_type: 'memory' }], counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 } })
    mockApiPost.mockResolvedValue({})
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(screen.getByText('web-scraper')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Run'))
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/collections/run?name=web-scraper')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Pipeline executed', 'success')
  })

  it('deletes a pipeline', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [{ id: 'p1', name: 'web-scraper', source_type: 'url', store_type: 'memory' }], counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 } })
    mockApiDelete.mockResolvedValue({})
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(screen.getByText('web-scraper')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => {
      expect(mockApiDelete).toHaveBeenCalledWith('/collections/p1')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Pipeline deleted', 'success')
  })

  it('shows error on fetch failure', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load collections', 'error')
    }, { timeout: 5000 })
  })

  it('refreshes on Refresh click', async () => {
    mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
    render(<CollectionsPage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
