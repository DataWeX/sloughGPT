import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { ConsciousnessChatPanel } from './ConsciousnessChatPanel'
import { emitConsciousness } from '@/lib/consciousness-bus'
import type { ConsciousnessEvent } from '@/lib/consciousness-bus'

vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => {
      const map: Record<string, string> = {
        'consciousness_chat.title': 'Consciousness',
        'consciousness_chat.current_state': 'Current State',
        'consciousness_chat.beliefs': 'Beliefs',
        'consciousness_chat.growth_delta': 'Growth Delta',
        'consciousness_chat.episode_info': 'Episode Info',
        'consciousness_chat.quick_actions': 'Quick Actions',
        'consciousness_chat.history': 'History',
        'consciousness_chat.no_data': 'No data yet',
        'consciousness_chat.no_episodes': 'No episodes',
        'consciousness_chat.no_history': 'No history',
        'consciousness_chat.reflect': 'Reflect',
        'consciousness_chat.reflecting': 'Reflecting...',
        'consciousness_chat.rate': 'Rate',
        'consciousness_chat.star': 'star',
        'consciousness_chat.seed_data': 'Seed Data',
        'consciousness_chat.seeding': 'Seeding...',
        'consciousness_chat.no_narrative': 'No narrative',
      }
      return map[key] ?? key
    },
  }),
}))

async function emitAndWait(event: ConsciousnessEvent) {
  await act(async () => {
    emitConsciousness(event)
  })
}

describe('ConsciousnessChatPanel', () => {
  const onClose = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) }))
  })

  it('returns null when not open', () => {
    const { container } = render(<ConsciousnessChatPanel open={false} onClose={onClose} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders panel when open', () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    expect(screen.getByText('Consciousness')).toBeDefined()
  })

  it('shows no-data message initially', () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    expect(screen.getByText('No data yet')).toBeDefined()
  })

  it('calls onClose on close button click', () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    const closeBtn = screen.getByRole('button', { name: /Close consciousness panel/i })
    fireEvent.click(closeBtn)
    expect(onClose).toHaveBeenCalled()
  })

  it('displays qualia bars when event received', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    await emitAndWait({
      level: 2,
      qualia: { joy: 0.8, curiosity: 0.6 },
      beliefs: {},
      growth_delta: 0.1,
      self_insight: 'test',
    })
    expect(screen.getByText('joy')).toBeDefined()
    expect(screen.getByText('curiosity')).toBeDefined()
  })

  it('shows level badge when event received', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    await emitAndWait({
      level: 3,
      qualia: {},
      beliefs: {},
      growth_delta: 0,
      self_insight: '',
    })
    expect(screen.getByText(/L3/)).toBeDefined()
  })

  it('expand/collapse toggles beliefs visibility', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    await emitAndWait({
      level: 1,
      qualia: { joy: 0.5 },
      beliefs: { growth: 0.9 },
      growth_delta: 0.05,
      self_insight: 'insight',
    })

    expect(screen.queryByText('growth')).toBeNull()

    const expandBtn = screen.getByRole('button', { name: /Expand panel/i })
    fireEvent.click(expandBtn)
    expect(screen.getByText('Beliefs')).toBeDefined()
  })

  it('reflect button calls POST /api/consciousness/reflect', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    const expandBtn = screen.getByRole('button', { name: /Expand panel/i })
    fireEvent.click(expandBtn)
    const reflectBtn = screen.getByRole('button', { name: /Reflect/i })
    fireEvent.click(reflectBtn)
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/consciousness/reflect', expect.objectContaining({ method: 'POST' }))
    })
  })

  it('seed button calls POST /api/consciousness/seed', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    const expandBtn = screen.getByRole('button', { name: /Expand panel/i })
    fireEvent.click(expandBtn)
    const seedBtn = screen.getByRole('button', { name: /Seed Data/i })
    fireEvent.click(seedBtn)
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/consciousness/seed', expect.objectContaining({ method: 'POST' }))
    })
  })

  it('star rating renders 5 stars', async () => {
    render(<ConsciousnessChatPanel open={true} onClose={onClose} />)
    const expandBtn = screen.getByRole('button', { name: /Expand panel/i })
    fireEvent.click(expandBtn)
    const stars = screen.getAllByRole('button').filter(b =>
      b.getAttribute('aria-label')?.includes('star')
    )
    expect(stars.length).toBe(5)
  })
})
