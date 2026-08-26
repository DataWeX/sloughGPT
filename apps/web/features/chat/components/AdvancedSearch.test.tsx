import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { AdvancedSearch } from './AdvancedSearch'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: Date.now() },
  { id: '2', role: 'assistant', content: 'Hi there! How can I help?', timestamp: Date.now() },
  { id: '3', role: 'user', content: 'Tell me about TypeScript', timestamp: Date.now() },
]

describe('AdvancedSearch', () => {
  it('renders search input', () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    expect(screen.getByPlaceholderText('Search messages...')).toBeInTheDocument()
  })

  it('finds matching messages', () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'hello' } })
    expect(screen.getByText('1/1')).toBeInTheDocument()
  })

  it('shows no matches for non-existent text', () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'xyz' } })
    expect(screen.getByText('No matches')).toBeInTheDocument()
  })

  it('toggles case sensitivity', () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'hello' } })
    expect(screen.getByText('1/1')).toBeInTheDocument()
    
    fireEvent.click(screen.getByTitle('Case sensitive'))
    expect(screen.getByText('No matches')).toBeInTheDocument()
  })

  it('navigates to next match', () => {
    const onHighlight = vi.fn()
    render(<AdvancedSearch messages={mockMessages} onHighlight={onHighlight} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'o' } })
    
    fireEvent.click(screen.getByLabelText('Next match'))
    expect(onHighlight).toHaveBeenCalled()
  })

  it('navigates to previous match', () => {
    const onHighlight = vi.fn()
    render(<AdvancedSearch messages={mockMessages} onHighlight={onHighlight} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'o' } })
    
    fireEvent.click(screen.getByLabelText('Previous match'))
    expect(onHighlight).toHaveBeenCalled()
  })

  it('clears search', () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'hello' } })
    expect(screen.getByText('1/1')).toBeInTheDocument()
    
    fireEvent.click(screen.getByLabelText('Clear search'))
    expect(screen.queryByText('1/1')).not.toBeInTheDocument()
  })

  it('toggles regex mode', async () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    
    await act(async () => {
      fireEvent.click(screen.getByTitle('Regex'))
    })
    
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'hel+' } })
    expect(screen.getByText(/\/\d+$/)).toBeInTheDocument()
    expect(screen.queryByText('No matches')).not.toBeInTheDocument()
  })

  it('toggles whole word mode', async () => {
    render(<AdvancedSearch messages={mockMessages} onHighlight={vi.fn()} />)
    
    await act(async () => {
      fireEvent.click(screen.getByTitle('Whole word'))
    })
    
    fireEvent.change(screen.getByPlaceholderText('Search messages...'), { target: { value: 'el' } })
    expect(screen.getByText('No matches')).toBeInTheDocument()
  })
})