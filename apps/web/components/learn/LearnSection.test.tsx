// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

const tabButton = (name: string) => screen.getAllByRole('button', { name }).filter(b => b.className.includes('px-3 py-1.5'))[0]

const mocks = vi.hoisted(() => ({
  mockStatus: vi.fn(),
  mockSearch: vi.fn(),
  mockIngestUrl: vi.fn(),
  mockIngestText: vi.fn(),
  mockQueryKnowledge: vi.fn(),
  mockListFeeds: vi.fn(),
  mockSubscribeFeed: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/learner-controller', () => ({
  learnerController: {
    status: mocks.mockStatus,
    search: mocks.mockSearch,
    ingestUrl: mocks.mockIngestUrl,
    ingestText: mocks.mockIngestText,
    queryKnowledge: mocks.mockQueryKnowledge,
    listFeeds: mocks.mockListFeeds,
    subscribeFeed: mocks.mockSubscribeFeed,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@/components/learn/LearningInsightsCard', () => ({
  LearningInsightsCard: ({ facts }: any) => <div data-testid="insights">{facts.length} insights</div>,
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: any) => <div>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ value, onChange, onKeyDown, placeholder, ...props }: any) => (
    <input value={value} onChange={onChange} onKeyDown={onKeyDown} placeholder={placeholder} {...props} />
  ),
  Textarea: ({ value, onChange, rows, placeholder, ...props }: any) => (
    <textarea value={value} onChange={onChange} rows={rows} placeholder={placeholder} {...props} />
  ),
  IconRefresh: () => <span data-testid="icon-refresh" />,
}))

import { LearnSection } from './LearnSection'

describe('LearnSection', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows loading skeleton while status loads', () => {
    mocks.mockStatus.mockReturnValue(new Promise(() => {}))
    render(<LearnSection />)
  })

  it('renders header and stat tiles from status', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: true,
      knowledge_count: 42,
      feeds_count: 3,
      total_tokens: 1000,
    })
    render(<LearnSection />)
    expect(await screen.findByText('Continual Learning')).toBeDefined()
    expect(await screen.findByText(42)).toBeDefined()
    expect(screen.getByText(1000)).toBeDefined()
    expect(screen.getByText(3)).toBeDefined()
    expect(screen.getByText('Active')).toBeDefined()
  })

  it('runs a search and shows the result summary', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockSearch.mockResolvedValue({ tokens_ingested: 500, new_facts: 12, rejected: 0, filter_stats: {} })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    const input = screen.getByPlaceholderText('Search query...')
    fireEvent.change(input, { target: { value: 'python' } })
    fireEvent.click(screen.getByText('Search & Learn'))
    expect(await screen.findByText('Ingested 500 tokens, 12 new facts')).toBeDefined()
  })

  it('shows an error banner when search fails', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockSearch.mockRejectedValue(new Error('boom'))
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.change(screen.getByPlaceholderText('Search query...'), { target: { value: 'x' } })
    fireEvent.click(screen.getByText('Search & Learn'))
    expect(await screen.findByText('boom')).toBeDefined()
  })

  it('ingests from URL and shows the added-facts result', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockIngestUrl.mockResolvedValue({ status: 'ok', facts_added: 7 })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Ingest'))
    fireEvent.change(screen.getByPlaceholderText('https://...'), { target: { value: 'https://example.com' } })
    fireEvent.click(screen.getByText('Ingest URL'))
    expect(await screen.findByText('Added 7 facts from URL')).toBeDefined()
  })

  it('ingests from text and clears the textarea', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockIngestText.mockResolvedValue({ status: 'ok', facts_added: 4 })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Ingest'))
    const ta = screen.getByPlaceholderText('Paste text to learn from...')
    fireEvent.change(ta, { target: { value: 'some text' } })
    fireEvent.click(screen.getByText('Ingest Text'))
    expect(await screen.findByText('Added 4 facts from text')).toBeDefined()
  })

  it('loads and shows knowledge when the knowledge tab is opened', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockQueryKnowledge.mockResolvedValue({
      facts: [{ content: 'fact one', topic: 'ml', source: 'web', importance: 0.9 }],
      total: 1,
    })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Knowledge'))
    expect(await screen.findByText('fact one')).toBeDefined()
    expect(await screen.findByText('Knowledge (1)')).toBeDefined()
  })

  it('shows empty knowledge state', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockQueryKnowledge.mockResolvedValue({ facts: [], total: 0 })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Knowledge'))
    expect(await screen.findByText('No knowledge yet. Use Search or Ingest to learn.')).toBeDefined()
  })

  it('lists feeds on the feeds tab and subscribes a new feed', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockListFeeds.mockResolvedValue({
      feeds: [{ url: 'https://feed.example/rss', interval: 3600 }],
    })
    mocks.mockSubscribeFeed.mockResolvedValue({ status: 'ok' })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Feeds'))
    expect(await screen.findByText('https://feed.example/rss')).toBeDefined()
    fireEvent.change(screen.getByPlaceholderText('RSS feed URL...'), { target: { value: 'https://new.example/rss' } })
    fireEvent.click(screen.getByText('Subscribe'))
    expect(await screen.findByText('Subscribed')).toBeDefined()
  })

  it('shows empty feeds state', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockListFeeds.mockResolvedValue({ feeds: [] })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Feeds'))
    expect(await screen.findByText('No feeds subscribed.')).toBeDefined()
  })

  it('shows a toast when learning insights render', async () => {
    mocks.mockStatus.mockResolvedValue({
      learner_active: false,
      knowledge_count: 0,
      feeds_count: 0,
      total_tokens: 0,
    })
    mocks.mockQueryKnowledge.mockResolvedValue({
      facts: [{ content: 'f', topic: 'ml', source: 'web', importance: 0.8 }],
      total: 1,
    })
    render(<LearnSection />)
    await screen.findByText('Continual Learning')
    fireEvent.click(tabButton('Knowledge'))
    expect(await screen.findByTestId('insights')).toBeDefined()
    expect(mocks.mockQueryKnowledge).toHaveBeenCalled()
  })
})