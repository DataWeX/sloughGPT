import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ChatAnalytics } from './ChatAnalytics'
import { chatDB } from '@/lib/db'

afterEach(cleanup)

vi.mock('@/lib/db', () => ({
  chatDB: {
    loadSessions: vi.fn(),
  },
}))

const mockSessions = [
  {
    id: '1',
    name: 'Chat 1',
    messages: [
      { role: 'user', content: 'Hello world' },
      { role: 'assistant', content: 'Hi there!' },
    ],
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-15T10:05:00Z',
  },
  {
    id: '2',
    name: 'Chat 2',
    messages: [
      { role: 'user', content: 'How are you?' },
      { role: 'assistant', content: 'I am good, thanks!' },
      { role: 'user', content: 'Great!' },
    ],
    createdAt: '2024-01-16T10:00:00Z',
    updatedAt: '2024-01-16T10:05:00Z',
  },
]

describe('ChatAnalytics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state initially', () => {
    vi.mocked(chatDB.loadSessions).mockImplementation(() => new Promise(() => {}))
    render(<ChatAnalytics />)
    expect(screen.getByText('Loading analytics...')).toBeInTheDocument()
  })

  it('shows no data when sessions are empty', async () => {
    vi.mocked(chatDB.loadSessions).mockResolvedValue([])
    render(<ChatAnalytics />)
    expect(await screen.findByText('No data available')).toBeInTheDocument()
  })

  it('renders analytics when data loaded', async () => {
    vi.mocked(chatDB.loadSessions).mockResolvedValue(mockSessions)
    render(<ChatAnalytics />)
    
    expect(await screen.findByText('2')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('shows session count', async () => {
    vi.mocked(chatDB.loadSessions).mockResolvedValue(mockSessions)
    render(<ChatAnalytics />)
    
    await screen.findByText('Sessions')
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('shows message count', async () => {
    vi.mocked(chatDB.loadSessions).mockResolvedValue(mockSessions)
    render(<ChatAnalytics />)
    
    await screen.findByText('Messages')
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('shows active days', async () => {
    vi.mocked(chatDB.loadSessions).mockResolvedValue(mockSessions)
    render(<ChatAnalytics />)
    
    expect(await screen.findByText('2 active days')).toBeInTheDocument()
  })

  it('handles error gracefully', async () => {
    vi.mocked(chatDB.loadSessions).mockRejectedValue(new Error('Failed'))
    render(<ChatAnalytics />)
    
    expect(await screen.findByText('No data available')).toBeInTheDocument()
  })
})