import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionCloning } from './ChatSessionCloning'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const makeMsg = (role: 'user' | 'assistant', content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  timestamp: new Date(),
})

const mockSessions = [
  {
    id: 's1',
    title: 'Session 1',
    messages: [makeMsg('user', 'Hello'), makeMsg('assistant', 'Hi')],
    createdAt: Date.now() - 86400000,
    lastActivity: Date.now(),
  },
  {
    id: 's2',
    title: 'Session 2',
    messages: [makeMsg('user', 'Test')],
    createdAt: Date.now() - 3600000,
    lastActivity: Date.now(),
  },
]

describe('ChatSessionCloning', () => {
  it('renders title', () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    expect(screen.getByText('Clone Session')).toBeInTheDocument()
  })

  it('renders session list', () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    expect(screen.getByText('Session 1')).toBeInTheDocument()
    expect(screen.getByText('Session 2')).toBeInTheDocument()
  })

  it('shows message count', () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    expect(screen.getByText('2 messages')).toBeInTheDocument()
    expect(screen.getByText('1 messages')).toBeInTheDocument()
  })

  it('shows empty state', () => {
    render(<ChatSessionCloning sessions={[]} onClone={vi.fn()} />)
    expect(screen.getByText('No sessions')).toBeInTheDocument()
  })

  it('opens confirm dialog on select', () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    fireEvent.click(screen.getByText('Session 1'))
    expect(screen.getByText(/Cloning/)).toBeInTheDocument()
    expect(screen.getByDisplayValue('Session 1 (Copy)')).toBeInTheDocument()
  })

  it('calls onClone with correct args', async () => {
    const onClone = vi.fn()
    render(<ChatSessionCloning sessions={mockSessions} onClone={onClone} />)
    fireEvent.click(screen.getByText('Session 1'))
    await act(async () => {
      fireEvent.click(screen.getByText('Clone'))
    })
    expect(onClone).toHaveBeenCalledWith('s1', 'Session 1 (Copy)')
  })

  it('allows editing title', async () => {
    const onClone = vi.fn()
    render(<ChatSessionCloning sessions={mockSessions} onClone={onClone} />)
    fireEvent.click(screen.getByText('Session 1'))
    fireEvent.change(screen.getByDisplayValue('Session 1 (Copy)'), { target: { value: 'My Clone' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Clone'))
    })
    expect(onClone).toHaveBeenCalledWith('s1', 'My Clone')
  })

  it('clones on Enter key', async () => {
    const onClone = vi.fn()
    render(<ChatSessionCloning sessions={mockSessions} onClone={onClone} />)
    fireEvent.click(screen.getByText('Session 1'))
    const input = screen.getByDisplayValue('Session 1 (Copy)')
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(onClone).toHaveBeenCalledWith('s1', 'Session 1 (Copy)')
  })

  it('cancels cloning', () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    fireEvent.click(screen.getByText('Session 1'))
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Cloning:')).not.toBeInTheDocument()
  })

  it('disables clone when title empty', async () => {
    render(<ChatSessionCloning sessions={mockSessions} onClone={vi.fn()} />)
    fireEvent.click(screen.getByText('Session 1'))
    fireEvent.change(screen.getByDisplayValue('Session 1 (Copy)'), { target: { value: '' } })
    const cloneBtn = screen.getByText('Clone').closest('button')!
    expect(cloneBtn).toBeDisabled()
  })
})