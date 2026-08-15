import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardContent: passthrough,
    Button: ({ children, onClick, disabled, variant, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} aria-label={ariaLabel}>{children}</button>
    ),
    Input: ({ value, onChange, onKeyDown, placeholder, className, type }: any) => (
      <input value={value} onChange={onChange} onKeyDown={onKeyDown} placeholder={placeholder} className={className} type={type} />
    ),
    Textarea: ({ value, onChange, placeholder, rows }: any) => (
      <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />
    ),
    IconRefresh: iconMock('refresh'),
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

const {
  mockStatus, mockSearch, mockIngestUrl, mockIngestText, mockQueryKnowledge, mockListFeeds, mockSubscribeFeed, mockAddToast,
} = vi.hoisted(() => ({
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
    status: mockStatus,
    search: mockSearch,
    ingestUrl: mockIngestUrl,
    ingestText: mockIngestText,
    queryKnowledge: mockQueryKnowledge,
    listFeeds: mockListFeeds,
    subscribeFeed: mockSubscribeFeed,
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))
vi.mock('@/components/learn/LearningInsightsCard', () => ({
  LearningInsightsCard: ({ facts }: any) => (facts.length > 0 ? <div data-testid="learning-insights-card" /> : null),
}))

import LearnPage from './page'

const status = {
  knowledge_count: 12,
  total_tokens: 3456,
  feeds_count: 3,
  learner_active: true,
}

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockStatus.mockResolvedValue(status)
})

async function renderLoaded() {
  render(<LearnPage />)
  await waitFor(() => { expect(screen.getByText('12 facts · 3456 tokens')).toBeTruthy() })
}

describe('LearnPage', () => {
  it('shows loading skeleton and calls status on mount', () => {
    mockStatus.mockReturnValue(new Promise(() => {}))
    render(<LearnPage />)
    expect(screen.getAllByText('Learner').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Continual web learning')).toBeTruthy()
    expect(mockStatus).toHaveBeenCalled()
  })

  it('displays status stats and header subtitle after load', async () => {
    render(<LearnPage />)
    await waitFor(() => { expect(screen.getByText('12 facts · 3456 tokens')).toBeTruthy() })
    expect(screen.getAllByText('Knowledge').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Tokens')).toBeTruthy()
    expect(screen.getAllByText('Feeds').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Status')).toBeTruthy()
    expect(screen.getAllByText('12').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3456').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('renders default search tab after status fails', async () => {
    mockStatus.mockRejectedValueOnce(new Error('boom'))
    render(<LearnPage />)
    await waitFor(() => { expect(screen.getByText('Search & Learn')).toBeTruthy() })
    expect(screen.getByText('Continual web learning')).toBeTruthy()
    expect(screen.queryByText('Status')).toBeFalsy()
  })

  it('keeps Search & Learn disabled and does not search with empty query', async () => {
    await renderLoaded()
    const btn = screen.getByText('Search & Learn') as HTMLButtonElement
    expect(btn.disabled).toBe(true)
    await act(async () => { btn.click() })
    expect(mockSearch).not.toHaveBeenCalled()
  })

  it('searches and displays ingested token/fact summary', async () => {
    mockSearch.mockResolvedValue({ tokens_ingested: 500, new_facts: 3, rejected: 0, filter_stats: {} })
    await renderLoaded()
    const input = screen.getByPlaceholderText('Search query...')
    await act(async () => { fireEvent.change(input, { target: { value: 'machine learning' } }) })
    await act(async () => { screen.getByText('Search & Learn').click() })
    await waitFor(() => { expect(mockSearch).toHaveBeenCalledWith('machine learning') })
    expect(screen.getByText('Ingested 500 tokens, 3 new facts')).toBeTruthy()
  })

  it('shows error message when search fails', async () => {
    mockSearch.mockRejectedValue(new Error('network down'))
    await renderLoaded()
    const input = screen.getByPlaceholderText('Search query...')
    await act(async () => { fireEvent.change(input, { target: { value: 'ml' } }) })
    await act(async () => { screen.getByText('Search & Learn').click() })
    await waitFor(() => { expect(screen.getByText('network down')).toBeTruthy() })
  })

  it('triggers search on Enter key', async () => {
    mockSearch.mockResolvedValue({ tokens_ingested: 10, new_facts: 1, rejected: 0, filter_stats: {} })
    await renderLoaded()
    const input = screen.getByPlaceholderText('Search query...')
    await act(async () => { fireEvent.change(input, { target: { value: 'news' } }) })
    await act(async () => { fireEvent.keyDown(input, { key: 'Enter' }) })
    await waitFor(() => { expect(mockSearch).toHaveBeenCalledWith('news') })
  })

  it('ingests URL and clears the field', async () => {
    mockIngestUrl.mockResolvedValue({ status: 'ok', facts_added: 7 })
    await renderLoaded()
    await act(async () => { screen.getByText('Ingest').click() })
    const urlInput = screen.getByPlaceholderText('https://...')
    await act(async () => { fireEvent.change(urlInput, { target: { value: 'https://example.com' } }) })
    await act(async () => { screen.getByText('Ingest URL').click() })
    await waitFor(() => { expect(mockIngestUrl).toHaveBeenCalledWith('https://example.com') })
    expect(screen.getByText('Added 7 facts from URL')).toBeTruthy()
    expect((screen.getByPlaceholderText('https://...') as HTMLInputElement).value).toBe('')
  })

  it('ingests pasted text', async () => {
    mockIngestText.mockResolvedValue({ status: 'ok', facts_added: 9 })
    await renderLoaded()
    await act(async () => { screen.getByText('Ingest').click() })
    const ta = screen.getByPlaceholderText('Paste text to learn from...')
    await act(async () => { fireEvent.change(ta, { target: { value: 'some article text' } }) })
    await act(async () => { screen.getByText('Ingest Text').click() })
    await waitFor(() => { expect(mockIngestText).toHaveBeenCalledWith('some article text') })
    expect(screen.getByText('Added 9 facts from text')).toBeTruthy()
  })

  it('shows error message when URL ingest fails', async () => {
    mockIngestUrl.mockRejectedValue(new Error('bad url'))
    await renderLoaded()
    await act(async () => { screen.getByText('Ingest').click() })
    const urlInput = screen.getByPlaceholderText('https://...')
    await act(async () => { fireEvent.change(urlInput, { target: { value: 'https://bad.example' } }) })
    await act(async () => { screen.getByText('Ingest URL').click() })
    await waitFor(() => { expect(screen.getByText('bad url')).toBeTruthy() })
  })

  it('loads knowledge facts and shows insights card', async () => {
    mockQueryKnowledge.mockResolvedValue({
      facts: [{ content: 'Paris is the capital of France', topic: 'geography', source: 'wikipedia', importance: 0.9 }],
      total: 1,
    })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Knowledge' }).click() })
    await waitFor(() => { expect(mockQueryKnowledge).toHaveBeenCalledWith(undefined) })
    expect(screen.getByText('Knowledge (1)')).toBeTruthy()
    expect(screen.getByText('Paris is the capital of France')).toBeTruthy()
    expect(screen.getByText('geography')).toBeTruthy()
    expect(screen.getByText('wikipedia')).toBeTruthy()
    expect(screen.getByTestId('learning-insights-card')).toBeTruthy()
  })

  it('shows empty knowledge state', async () => {
    mockQueryKnowledge.mockResolvedValue({ facts: [], total: 0 })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Knowledge' }).click() })
    await waitFor(() => { expect(screen.getByText('No knowledge yet. Use Search or Ingest to learn.')).toBeTruthy() })
    expect(screen.queryByTestId('learning-insights-card')).toBeFalsy()
  })

  it('filters knowledge by topic via Search button', async () => {
    mockQueryKnowledge.mockResolvedValue({
      facts: [{ content: 'Water freezes at 0C', topic: 'science', source: 'web', importance: 0.5 }],
      total: 1,
    })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Knowledge' }).click() })
    await waitFor(() => { expect(mockQueryKnowledge).toHaveBeenCalledWith(undefined) })
    const filter = screen.getByPlaceholderText('Filter by topic...')
    await act(async () => { fireEvent.change(filter, { target: { value: 'science' } }) })
    const searchButtons = screen.getAllByText('Search')
    await act(async () => { searchButtons[1].click() })
    await waitFor(() => { expect(mockQueryKnowledge).toHaveBeenCalledWith('science') })
  })

  it('shows error toast when knowledge load fails', async () => {
    mockQueryKnowledge.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Knowledge' }).click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to load knowledge', 'error') })
  })

  it('lists subscribed RSS feeds', async () => {
    mockListFeeds.mockResolvedValue({ feeds: [{ url: 'https://news.example.com/rss', interval: 3600, last_poll: '2026-08-01T10:00:00Z' }] })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Feeds' }).click() })
    await waitFor(() => { expect(mockListFeeds).toHaveBeenCalled() })
    expect(screen.getByText('RSS Feeds')).toBeTruthy()
    expect(screen.getByText('https://news.example.com/rss')).toBeTruthy()
    expect(screen.getByText('3600s')).toBeTruthy()
  })

  it('shows empty feeds state', async () => {
    mockListFeeds.mockResolvedValue({ feeds: [] })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Feeds' }).click() })
    await waitFor(() => { expect(screen.getByText('No feeds subscribed.')).toBeTruthy() })
  })

  it('subscribes to a feed and refreshes the list', async () => {
    mockListFeeds.mockResolvedValue({ feeds: [] })
    mockSubscribeFeed.mockResolvedValue({ status: 'ok' })
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Feeds' }).click() })
    await waitFor(() => { expect(mockListFeeds).toHaveBeenCalledTimes(1) })
    const feedInput = screen.getByPlaceholderText('RSS feed URL...')
    await act(async () => { fireEvent.change(feedInput, { target: { value: 'https://rss.example.com' } }) })
    await act(async () => { screen.getByText('Subscribe').click() })
    await waitFor(() => { expect(mockSubscribeFeed).toHaveBeenCalledWith('https://rss.example.com') })
    expect(screen.getByText('Subscribed')).toBeTruthy()
    expect(mockListFeeds).toHaveBeenCalledTimes(2)
  })

  it('shows Failed message when subscribing fails', async () => {
    mockListFeeds.mockResolvedValue({ feeds: [] })
    mockSubscribeFeed.mockRejectedValue(new Error('boom'))
    await renderLoaded()
    await act(async () => { screen.getByRole('button', { name: 'Feeds' }).click() })
    await waitFor(() => { expect(mockListFeeds).toHaveBeenCalledTimes(1) })
    const feedInput = screen.getByPlaceholderText('RSS feed URL...')
    await act(async () => { fireEvent.change(feedInput, { target: { value: 'https://rss.example.com' } }) })
    await act(async () => { screen.getByText('Subscribe').click() })
    await waitFor(() => { expect(screen.getByText('Failed')).toBeTruthy() })
  })
})
