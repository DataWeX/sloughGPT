// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/components/ui', () => ({
  IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
  IconCheck: () => <span data-testid="icon-check">check</span>,
  IconX: () => <span data-testid="icon-x">x</span>,
}))

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, variant, size, ...rest }: any) => (
    <button onClick={onClick} disabled={disabled} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
}))

import { ContextInspector } from './ContextInspector'

function makeInspectorData(overrides = {}) {
  return {
    session: { id: 's1', message_count: 5, messages: [] },
    knowledge: { total_facts: 0, topics: [] },
    traits: {},
    modes: {},
    feedback: { total: 0, thumbs_up: 0, thumbs_down: 0 },
    workspace: { working_memory: [], semantic_keys: [], episodic_count: 0, sensory_buffer_size: 0, system_prompt: '' },
    ...overrides,
  }
}

describe('ContextInspector', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('shows no active session when sessionId is null', () => {
    render(<ContextInspector sessionId={null} />)
    expect(screen.getByText('No active session')).toBeDefined()
  })

  it('shows loading skeleton on mount', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    const { container } = render(<ContextInspector sessionId="s1" />)
    expect(container.querySelector('.animate-pulse')).toBeDefined()
    vi.unstubAllGlobals()
  })

  it('shows error with retry button on fetch failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 500 })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('HTTP 500')).toBeDefined()
    })
    expect(screen.getByText('Retry')).toBeDefined()
    vi.unstubAllGlobals()
  })

  it('renders session message count', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData()),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('5 messages')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders system prompt', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        workspace: { working_memory: [], semantic_keys: [], episodic_count: 0, sensory_buffer_size: 0, system_prompt: 'You are helpful.' },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('System Prompt')).toBeDefined()
      expect(screen.getByText('You are helpful.')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders manager modes', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        modes: {
          personality: { label: 'Warm', confidence: 0.85, scores: { warmth: 0.85 } },
          memory: { label: 'Detail', confidence: 0.6, capacity: 8, scores: { detail: 0.6 } },
        },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('Active Modes')).toBeDefined()
      expect(screen.getByText(/Personality: Warm/)).toBeDefined()
      expect(screen.getByText(/Memory: Detail/)).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders trait bars', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        traits: { personality: { warmth: 0.9, creativity: 0.7 } },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('Trait Weights')).toBeDefined()
      expect(screen.getByText('warmth')).toBeDefined()
      expect(screen.getByText('creativity')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders knowledge section when facts > 0', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        knowledge: { total_facts: 3, topics: ['python', 'testing'] },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('Knowledge Base')).toBeDefined()
      expect(screen.getByText('3 facts')).toBeDefined()
      expect(screen.getByText('python')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders feedback section when total > 0', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        feedback: { total: 10, thumbs_up: 7, thumbs_down: 3 },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('Feedback')).toBeDefined()
      expect(screen.getByText('7')).toBeDefined()
      expect(screen.getByText('3')).toBeDefined()
      expect(screen.getByText('10 total')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('renders workspace memory when semantic_keys present', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(makeInspectorData({
        workspace: { working_memory: [], semantic_keys: ['key1', 'key2'], episodic_count: 4, sensory_buffer_size: 0, system_prompt: '' },
      })),
    })))
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => {
      expect(screen.getByText('Workspace Memory')).toBeDefined()
      expect(screen.getByText('4 episodes')).toBeDefined()
      expect(screen.getByText('key1')).toBeDefined()
    })
    vi.unstubAllGlobals()
  })

  it('retries fetch on Retry button click', async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error('Fail'))
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(makeInspectorData()) })
    vi.stubGlobal('fetch', fetchMock)
    render(<ContextInspector sessionId="s1" />)
    await waitFor(() => { expect(screen.getByText('Fail')).toBeDefined() })
    fireEvent.click(screen.getByText('Retry'))
    await waitFor(() => {
      expect(screen.getByText('5 messages')).toBeDefined()
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    vi.unstubAllGlobals()
  })
})
