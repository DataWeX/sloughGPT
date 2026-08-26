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
      { id: 'm1', role: 'user' as const, content: 'Hello world', timestamp: new Date() },
      { id: 'm2', role: 'assistant' as const, content: 'Hi there!', timestamp: new Date() },
    ],
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-15T10:05:00Z',
    synced: true,
    starred: false,
    pinned: false,
  },
  {
    id: '2',
    name: 'Chat 2',
    messages: [
      { id: 'm3', role: 'user' as const, content: 'How are you?', timestamp: new Date() },
      { id: 'm4', role: 'assistant' as const, content: 'I am good, thanks!', timestamp: new Date() },
      { id: 'm5', role: 'user' as const, content: 'Great!', timestamp: new Date() },
    ],
    createdAt: '2024-01-16T10:00:00Z',
    updatedAt: '2024-01-16T10:05:00Z',
    synced: true,
    starred: false,
    pinned: false,
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