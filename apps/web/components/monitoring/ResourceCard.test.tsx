import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ResourceCard } from './ResourceCard'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
    Card: passthrough, CardContent: passthrough,
    StatCard: ({ label, value }: any) => (
      <div data-testid={`stat-${label}`}>
        {typeof value === 'string' || typeof value === 'number' ? <span>{String(value)}</span> : value}
      </div>
    ),
    KpiGrid: ({ children }: any) => <div data-testid="kpi-grid">{children}</div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    StatusDot: () => <span data-testid="status-dot" />,
  }
})

vi.mock('@/components/composed/SectionLabel', () => ({
  SectionLabel: ({ children }: any) => <div>{children}</div>,
}))

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
    const skeletons = document.querySelectorAll('[data-testid="skeleton"]')
    expect(skeletons.length).toBeGreaterThanOrEqual(2)
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
    expect(screen.getByText(/2\.5 \/ 8\.0 GB/)).toBeDefined()
  })

  it('shows available memory when detailed health provides it', () => {
    renderCard({ detailed, metrics })
    expect(screen.getByText(/free/)).toBeDefined()
  })

  it('prefers liveHealth over metrics when both present', () => {
    renderCard({ liveHealth, metrics })
    expect(screen.getByText('55%')).toBeDefined()
    expect(screen.getByText('66%')).toBeDefined()
  })
})
