import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ResourceCard } from './ResourceCard'

const detailed = {
  system: { memory_available_mb: 8192 },
} as any

const metrics = { cpu_percent: 33, memory_percent: 44, memory_used_gb: 2.5, memory_total_gb: 8 } as any

const liveHealth = { cpu_percent: 55, memory_percent: 66 } as any

function renderCard(props: Partial<Parameters<typeof ResourceCard>[0]> = {}) {
  const base = {
    liveHealth: null,
    metrics: null,
    detailed: null,
    cpuThreshold: 80,
    memThreshold: 80,
    loaded: false,
  }
  return render(<ResourceCard {...base} {...props} />)
}

describe('ResourceCard', () => {
  afterEach(cleanup)

  it('shows skeleton placeholders when no data', () => {
    renderCard()
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(4)
  })

  it('renders cpu and memory from liveHealth', () => {
    renderCard({ liveHealth })
    expect(screen.getByText('55%')).toBeDefined()
    expect(screen.getByText('66%')).toBeDefined()
  })

  it('falls back to metrics when liveHealth is missing', () => {
    renderCard({ metrics })
    expect(screen.getByText('33%')).toBeDefined()
    expect(screen.getByText('44%')).toBeDefined()
  })

  it('renders used memory from metrics', () => {
    renderCard({ metrics })
    expect(screen.getByText('2.5 GB')).toBeDefined()
  })

  it('renders available memory from detailed health', () => {
    renderCard({ detailed })
    expect(screen.getByText('8.0 GB')).toBeDefined()
  })

  it('prefers liveHealth over metrics when both present', () => {
    renderCard({ liveHealth, metrics })
    expect(screen.getByText('55%')).toBeDefined()
    expect(screen.getByText('66%')).toBeDefined()
  })
})
