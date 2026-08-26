import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { SessionComparison } from './SessionComparison'
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
    messages: [makeMsg('user', 'Hello'), makeMsg('assistant', 'Hi there')],
    createdAt: Date.now() - 86400000,
    lastActivity: Date.now(),
  },
  {
    id: 's2',
    title: 'Session 2',
    messages: [makeMsg('user', 'Hello'), makeMsg('assistant', 'Hey!')],
    createdAt: Date.now() - 3600000,
    lastActivity: Date.now(),
  },
]

describe('SessionComparison', () => {
  it('renders comparison title', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    expect(screen.getByText('Session Comparison')).toBeInTheDocument()
  })

  it('shows message about needing 2 sessions', () => {
    render(<SessionComparison sessions={[mockSessions[0]]} onClose={vi.fn()} />)
    expect(screen.getByText(/Need at least 2 sessions/)).toBeInTheDocument()
  })

  it('renders session selectors', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    expect(screen.getByDisplayValue('Session 1')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Session 2')).toBeInTheDocument()
  })

  it('shows diff stats', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    expect(screen.getByText(/\+0/)).toBeInTheDocument()
    expect(screen.getByText(/-0/)).toBeInTheDocument()
  })

  it('calls onClose when close clicked', () => {
    const onClose = vi.fn()
    render(<SessionComparison sessions={mockSessions} onClose={onClose} />)
    fireEvent.click(screen.getByLabelText('Close comparison'))
    expect(onClose).toHaveBeenCalled()
  })

  it('toggles show unchanged', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
  })

  it('changes session selection', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    const select = screen.getAllByRole('combobox')[0]
    fireEvent.change(select, { target: { value: 's2' } })
    expect(screen.getAllByDisplayValue('Session 2')).toHaveLength(2)
  })

  it('shows modified content', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    expect(screen.getByText('Hi there')).toBeInTheDocument()
    expect(screen.getByText('Hey!')).toBeInTheDocument()
  })

  it('shows identical message', () => {
    render(<SessionComparison sessions={mockSessions} onClose={vi.fn()} />)
    const hellos = screen.getAllByText('Hello')
    expect(hellos.length).toBeGreaterThanOrEqual(1)
  })
})