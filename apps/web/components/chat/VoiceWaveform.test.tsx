import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { VoiceWaveform, VoiceOrb, ListeningIndicator, ListeningBars } from './VoiceWaveform'

describe('VoiceWaveform', () => {
  afterEach(cleanup)

  it('renders correct number of bars', () => {
    const { container } = render(<VoiceWaveform level={0.5} bars={12} />)
    const bars = container.querySelectorAll('[role="img"] > div')
    expect(bars.length).toBe(12)
  })

  it('applies mic variant color', () => {
    const { container } = render(<VoiceWaveform level={0.5} variant="mic" />)
    const bar = container.querySelector('.bg-primary')
    expect(bar).toBeTruthy()
  })

  it('applies speaker variant color', () => {
    const { container } = render(<VoiceWaveform level={0.5} variant="speaker" />)
    const bar = container.querySelector('.bg-emerald-500')
    expect(bar).toBeTruthy()
  })

  it('applies idle variant color', () => {
    const { container } = render(<VoiceWaveform level={0} variant="idle" />)
    const bar = container.querySelector('.bg-muted-foreground\\/30')
    expect(bar).toBeTruthy()
  })

  it('has accessible label', () => {
    render(<VoiceWaveform level={0.5} variant="mic" />)
    const el = document.querySelector('[role="img"]')
    expect(el?.getAttribute('aria-label')).toBe('Audio mic waveform')
  })

  it('scales bar height with level', () => {
    const { container, rerender } = render(<VoiceWaveform level={0.1} bars={4} height={100} />)
    const bars = container.querySelectorAll('[role="img"] > div')
    const lowHeight = bars[0].getAttribute('style')

    rerender(<VoiceWaveform level={0.9} bars={4} height={100} />)
    const highBars = container.querySelectorAll('[role="img"] > div')
    const highHeight = highBars[0].getAttribute('style')

    // Higher level should produce taller bars (higher min-height in px)
    expect(lowHeight).not.toBe(highHeight)
  })

  it('applies custom dimensions', () => {
    const { container } = render(<VoiceWaveform level={0.5} width={300} height={100} />)
    const el = container.querySelector('[role="img"]')
    expect(el?.getAttribute('style')).toContain('width: 300px')
    expect(el?.getAttribute('style')).toContain('height: 100px')
  })
})

describe('VoiceOrb', () => {
  afterEach(cleanup)

  it('renders children', () => {
    render(
      <VoiceOrb state="idle" micLevel={0}>
        <button>Click me</button>
      </VoiceOrb>
    )
    expect(screen.getByText('Click me')).toBeDefined()
  })

  it('shows ping animation when speaking', () => {
    const { container } = render(
      <VoiceOrb state="speaking" micLevel={0}>
        <span>child</span>
      </VoiceOrb>
    )
    const pings = container.querySelectorAll('.animate-ping')
    expect(pings.length).toBeGreaterThan(0)
  })

  it('shows spin animation when processing', () => {
    const { container } = render(
      <VoiceOrb state="processing" micLevel={0}>
        <span>child</span>
      </VoiceOrb>
    )
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeTruthy()
  })

  it('scales orb based on mic level when listening', () => {
    const { container, rerender } = render(
      <VoiceOrb state="listening" micLevel={0}>
        <span>child</span>
      </VoiceOrb>
    )
    const orb = container.querySelector('.relative.z-10')
    const lowScale = orb?.getAttribute('style')

    rerender(
      <VoiceOrb state="listening" micLevel={0.8}>
        <span>child</span>
      </VoiceOrb>
    )
    const highScale = orb?.getAttribute('style')
    expect(lowScale).not.toBe(highScale)
  })

  it('applies error border color on error state', () => {
    const { container } = render(
      <VoiceOrb state="error" micLevel={0}>
        <span>child</span>
      </VoiceOrb>
    )
    const orb = container.querySelector('.border-red-500')
    expect(orb).toBeTruthy()
  })
})

describe('ListeningIndicator', () => {
  afterEach(cleanup)

  it('renders nothing when inactive', () => {
    const { container } = render(<ListeningIndicator micLevel={0.5} active={false} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders rings when active', () => {
    const { container } = render(<ListeningIndicator micLevel={0.5} active={true} />)
    const rings = container.querySelectorAll('.rounded-full.border.border-primary\\/20')
    expect(rings.length).toBe(5)
  })

  it('renders center glow when active', () => {
    const { container } = render(<ListeningIndicator micLevel={0.5} active={true} />)
    const glow = container.querySelector('.bg-primary\\/5')
    expect(glow).toBeTruthy()
  })

  it('scales rings with mic level', () => {
    const { container, rerender } = render(<ListeningIndicator micLevel={0.1} active={true} />)
    const rings = container.querySelectorAll('.rounded-full.border.border-primary\\/20')
    const lowStyle = rings[0]?.getAttribute('style')

    rerender(<ListeningIndicator micLevel={0.9} active={true} />)
    const highRings = container.querySelectorAll('.rounded-full.border.border-primary\\/20')
    const highStyle = highRings[0]?.getAttribute('style')

    expect(lowStyle).not.toBe(highStyle)
  })
})

describe('ListeningBars', () => {
  afterEach(cleanup)

  it('renders nothing when inactive', () => {
    const { container } = render(<ListeningBars micLevel={0.5} active={false} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders correct number of bars', () => {
    const { container } = render(<ListeningBars micLevel={0.5} active={true} barCount={5} />)
    const bars = container.querySelectorAll('.w-1.rounded-full.bg-primary\\/60')
    expect(bars.length).toBe(5)
  })

  it('has accessible label', () => {
    render(<ListeningBars micLevel={0.5} active={true} />)
    const el = screen.getByLabelText('Listening animation')
    expect(el).toBeDefined()
  })

  it('scales bar height with mic level', () => {
    const { container, rerender } = render(<ListeningBars micLevel={0.1} active={true} barCount={3} />)
    const bars = container.querySelectorAll('.w-1.rounded-full.bg-primary\\/60')
    const lowHeight = bars[0]?.getAttribute('style')

    rerender(<ListeningBars micLevel={0.9} active={true} barCount={3} />)
    const highBars = container.querySelectorAll('.w-1.rounded-full.bg-primary\\/60')
    const highHeight = highBars[0]?.getAttribute('style')

    expect(lowHeight).not.toBe(highHeight)
  })
})
