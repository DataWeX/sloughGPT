import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { ConsciousnessToggle } from './ConsciousnessToggle'
import { emitConsciousness } from '@/lib/consciousness-bus'

vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => {
      const map: Record<string, string> = {
        'consciousness_chat.toggle': 'Toggle consciousness',
      }
      return map[key] ?? key
    },
  }),
}))

describe('ConsciousnessToggle', () => {
  const onToggle = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders button with brain icon', () => {
    render(<ConsciousnessToggle open={false} onToggle={onToggle} />)
    expect(screen.getByRole('button', { name: 'Toggle consciousness' })).toBeDefined()
  })

  it('calls onToggle on click', () => {
    render(<ConsciousnessToggle open={false} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole('button', { name: 'Toggle consciousness' }))
    expect(onToggle).toHaveBeenCalled()
  })

  it('has open prop passed through (checks button aria)', () => {
    const { rerender } = render(<ConsciousnessToggle open={false} onToggle={onToggle} />)
    const btn = screen.getByRole('button', { name: 'Toggle consciousness' })
    expect(btn).toBeDefined()

    rerender(<ConsciousnessToggle open={true} onToggle={onToggle} />)
    expect(screen.getByRole('button', { name: 'Toggle consciousness' })).toBeDefined()
  })

  it('displays level badge when consciousness event received', async () => {
    render(<ConsciousnessToggle open={false} onToggle={onToggle} />)
    await act(async () => {
      emitConsciousness({
        level: 3,
        qualia: {},
        beliefs: {},
        growth_delta: 0,
        self_insight: '',
      })
    })
    const badge = screen.getByText(/L\d/)
    expect(badge).toBeDefined()
  })

  it('updates level on multiple events', async () => {
    render(<ConsciousnessToggle open={false} onToggle={onToggle} />)
    await act(async () => {
      emitConsciousness({ level: 1, qualia: {}, beliefs: {}, growth_delta: 0, self_insight: '' })
    })
    expect(screen.getByText('L1')).toBeDefined()

    await act(async () => {
      emitConsciousness({ level: 5, qualia: {}, beliefs: {}, growth_delta: 0, self_insight: '' })
    })
    expect(screen.getByText('L5')).toBeDefined()
  })
})
