// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { ComponentHealthStrip, quantSummary } from './ComponentHealthStrip'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

afterEach(cleanup)

function makeHealth(overrides: Partial<LiveHealthSnapshot> = {}): LiveHealthSnapshot {
  return {
    model_loaded: true,
    model_loading: false,
    model_type: 'SloughGPT-1b',
    quantization: { bits: 4 },
    inference_count: 12,
    health_score: 92,
    health_status: 'healthy',
    cpu_percent: 12.5,
    memory_percent: 44.2,
    ...overrides,
  } as unknown as LiveHealthSnapshot
}

describe('quantSummary', () => {
  it('summarizes known quantization shapes', () => {
    expect(quantSummary({ bits: 4 })).toBe('4-bit')
    expect(quantSummary('q4_k_m')).toBe('q4_k_m')
    expect(quantSummary(8)).toBe('8-bit')
    expect(quantSummary({ method: 'awq' })).toBe('awq')
  })

  it('never dumps the raw object', () => {
    expect(quantSummary({ some: { nested: true } })).toBe('on')
    expect(quantSummary(null)).toBeNull()
    expect(quantSummary(undefined)).toBeNull()
  })
})

describe('ComponentHealthStrip', () => {
  it('labels inference and engines/system components', () => {
    render(<ComponentHealthStrip health={makeHealth()} connectionStatus="connected" />)
    expect(screen.getByText('Inference')).toBeInTheDocument()
    expect(screen.getByText('Engines / system')).toBeInTheDocument()
  })

  it('shows model, quantization, health score and resources', () => {
    render(<ComponentHealthStrip health={makeHealth()} connectionStatus="connected" />)
    expect(screen.getByText('SloughGPT-1b')).toBeInTheDocument()
    expect(screen.getByText('4-bit')).toBeInTheDocument()
    expect(screen.getByText('92/100')).toBeInTheDocument()
    expect(screen.getByText('12.5%')).toBeInTheDocument()
    expect(screen.getByText('44.2%')).toBeInTheDocument()
  })

  it('reports a missing model as needing setup', () => {
    render(
      <ComponentHealthStrip
        health={makeHealth({ model_loaded: false, model_type: null, quantization: null })}
        connectionStatus="connected"
      />,
    )
    expect(screen.getByText('not loaded')).toBeInTheDocument()
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.getAllByText('no model').length).toBeGreaterThanOrEqual(1)
  })

  it('shows skeletons while health is unavailable', () => {
    render(<ComponentHealthStrip health={null} />)
    expect(screen.getByTestId('component-health-strip')).toBeInTheDocument()
    expect(screen.queryAllByTestId('component-cell')).toHaveLength(0)
  })

  it('uses design tokens only — no raw colors', () => {
    const { container } = render(
      <ComponentHealthStrip health={makeHealth()} connectionStatus="connected" />,
    )
    expect(container.innerHTML).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
    expect(container.innerHTML).not.toMatch(/rgba?\(/)
  })
})
