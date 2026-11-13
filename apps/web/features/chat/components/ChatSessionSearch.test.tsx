import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionSearch } from './ChatSessionSearch'

afterEach(cleanup)

const mockSessions = [
  { id: 's1', title: 'Alpha Chat', messageCount: 10, createdAt: 100, updatedAt: 400 },
  { id: 's2', title: 'Beta Discussion', messageCount: 25, createdAt: 200, updatedAt: 300 },
  { id: 's3', title: 'Gamma Talk', messageCount: 5, createdAt: 300, updatedAt: 200 },
]

describe('ChatSessionSearch', () => {
  it('renders search input', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    expect(screen.getByPlaceholderText('Search sessions...')).toBeInTheDocument()
  })

  it('renders tabs', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    expect(screen.getByText('Search')).toBeInTheDocument()
    expect(screen.getByText('Sort')).toBeInTheDocument()
  })

  it('filters sessions by query', () => {
    const onFiltered = vi.fn()
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={onFiltered} />)
    fireEvent.change(screen.getByPlaceholderText('Search sessions...'), {
      target: { value: 'Alpha' },
    })
    expect(onFiltered).toHaveBeenCalledWith(
      expect.arrayContaining([expect.objectContaining({ title: 'Alpha Chat' })])
    )
  })

  it('shows result count', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search sessions...'), {
      target: { value: 'Alpha' },
    })
    expect(screen.getByText(/1 of 3 sessions/)).toBeInTheDocument()
  })

  it('shows empty state', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search sessions...'), {
      target: { value: 'Nonexistent' },
    })
    expect(screen.getByText(/No sessions match/)).toBeInTheDocument()
  })

  it('clears search on empty input', () => {
    const onFiltered = vi.fn()
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={onFiltered} />)
    fireEvent.change(screen.getByPlaceholderText('Search sessions...'), {
      target: { value: 'Alpha' },
    })
    fireEvent.change(screen.getByPlaceholderText('Search sessions...'), {
      target: { value: '' },
    })
    expect(onFiltered).toHaveBeenCalledWith(mockSessions)
  })

  it('switches to sort tab', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    fireEvent.click(screen.getByText('Sort'))
    expect(screen.getByText('Last Updated')).toBeInTheDocument()
    expect(screen.getByText('Date Created')).toBeInTheDocument()
    expect(screen.getByText('Title')).toBeInTheDocument()
    expect(screen.getByText('Messages')).toBeInTheDocument()
  })

  it('sorts by title', () => {
    const onFiltered = vi.fn()
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={onFiltered} />)
    fireEvent.click(screen.getByText('Sort'))
    fireEvent.click(screen.getByText('Title'))
    expect(onFiltered).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ title: 'Alpha Chat' }),
        expect.objectContaining({ title: 'Beta Discussion' }),
        expect.objectContaining({ title: 'Gamma Talk' }),
      ])
    )
  })

  it('reverses sort direction on second click', () => {
    const onFiltered = vi.fn()
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={onFiltered} />)
    fireEvent.click(screen.getByText('Sort'))
    fireEvent.click(screen.getByText('Title'))
    fireEvent.click(screen.getByText('Title'))
    expect(onFiltered).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ title: 'Gamma Talk' }),
        expect.objectContaining({ title: 'Beta Discussion' }),
        expect.objectContaining({ title: 'Alpha Chat' }),
      ])
    )
  })

  it('sorts by messages', () => {
    const onFiltered = vi.fn()
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={onFiltered} />)
    fireEvent.click(screen.getByText('Sort'))
    fireEvent.click(screen.getByText('Messages'))
    expect(onFiltered).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ title: 'Beta Discussion' }),
        expect.objectContaining({ title: 'Alpha Chat' }),
        expect.objectContaining({ title: 'Gamma Talk' }),
      ])
    )
  })

  it('switches back to search tab', () => {
    render(<ChatSessionSearch sessions={mockSessions} onFiltered={vi.fn()} />)
    fireEvent.click(screen.getByText('Sort'))
    fireEvent.click(screen.getByText('Search'))
    expect(screen.getByPlaceholderText('Search sessions...')).toBeInTheDocument()
  })
})