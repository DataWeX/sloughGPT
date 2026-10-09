import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ConsciousnessMessageBadge } from './ConsciousnessMessageBadge'
import { onConsciousness, emitConsciousness } from '@/lib/consciousness-bus'
import { getQualiaMood } from '@/hooks/useConsciousnessStatus'

vi.mock('@/hooks/useConsciousnessStatus', () => ({
  getQualiaMood: vi.fn().mockReturnValue('Happy'),
}))

vi.mock('@/lib/consciousness-bus', () => ({
  onConsciousness: vi.fn().mockReturnValue(() => {}),
  emitConsciousness: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
}))

describe('ConsciousnessMessageBadge', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns null when no event received', () => {
    const { container } = render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(container.innerHTML).toBe('')
  })

  it('subscribes to consciousness events', () => {
    render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(onConsciousness).toHaveBeenCalled()
  })

  it('renders mood when event received', () => {
    vi.mocked(onConsciousness).mockImplementation((callback: any) => {
      callback({
        messageId: 'msg-1',
        qualia: { valence: 0.5 },
        growth_delta: 0.05,
        level: 2,
      })
      return () => {}
    })
    render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(screen.getByText('Happy')).toBeTruthy()
  })

  it('renders growth delta', () => {
    vi.mocked(onConsciousness).mockImplementation((callback: any) => {
      callback({
        messageId: 'msg-1',
        qualia: {},
        growth_delta: 0.05,
        level: 2,
      })
      return () => {}
    })
    render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(screen.getByText('+0.050')).toBeTruthy()
  })

  it('renders negative growth in red', () => {
    vi.mocked(onConsciousness).mockImplementation((callback: any) => {
      callback({
        messageId: 'msg-1',
        qualia: {},
        growth_delta: -0.03,
        level: 2,
      })
      return () => {}
    })
    render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(screen.getByText('-0.030')).toBeTruthy()
  })

  it('renders level', () => {
    vi.mocked(onConsciousness).mockImplementation((callback: any) => {
      callback({
        messageId: 'msg-1',
        qualia: {},
        growth_delta: 0,
        level: 3,
      })
      return () => {}
    })
    render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(screen.getByText('L3')).toBeTruthy()
  })

  it('does not render growth when zero', () => {
    vi.mocked(onConsciousness).mockImplementation((callback: any) => {
      callback({
        messageId: 'msg-1',
        qualia: {},
        growth_delta: 0,
        level: 2,
      })
      return () => {}
    })
    const { container } = render(<ConsciousnessMessageBadge messageId="msg-1" />)
    expect(container.textContent).not.toContain('+')
    expect(container.textContent).not.toContain('-0.')
  })
})
