// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { render, cleanup } from '@testing-library/react'
import React from 'react'

import { StatTileGrid } from './stat-tile'

describe('StatTileGrid', () => {
  afterEach(cleanup)

  const tiles = [
    { key: 'step', label: 'Step', value: '42', hint: '/100' },
    { key: 'loss', label: 'Loss', value: '1.5000' },
    { key: 'speed', label: 'Speed', value: '2.5', hint: ' steps/s' },
  ]

  it('renders every tile with its label, value and hint', () => {
    const { getByText } = render(<StatTileGrid tiles={tiles} />)
    expect(getByText('Step')).toBeDefined()
    expect(getByText('42')).toBeDefined()
    expect(getByText('/100')).toBeDefined()
    expect(getByText('Loss')).toBeDefined()
    expect(getByText('1.5000')).toBeDefined()
    expect(getByText('Speed')).toBeDefined()
    expect(getByText('2.5')).toBeDefined()
    expect(getByText(/steps\/s/)).toBeDefined()
  })

  it('renders tiles without a hint without an empty hint span', () => {
    const { getByText } = render(
      <StatTileGrid tiles={[{ key: 'loss', label: 'Loss', value: '0.3000' }]} />,
    )
    expect(getByText('Loss')).toBeDefined()
    expect(getByText('0.3000')).toBeDefined()
  })

  it('renders nothing when there are no tiles', () => {
    const { container } = render(<StatTileGrid tiles={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('uses a gap-compensated flex layout so partial rows fill without holes', () => {
    const { container } = render(<StatTileGrid tiles={tiles} />)
    const row = container.firstChild as HTMLElement
    expect(row.className).toContain('flex')
    expect(row.className).toContain('flex-wrap')
    expect(row.className).toContain('gap-2')
    // per-item basis is gap-compensated (50% mobile / 25% sm) — a plain
    // percentage grid leaves an empty cell when items are not a full row
    const firstTile = row.firstElementChild as HTMLElement
    expect(firstTile.className).toContain('grow')
    expect(firstTile.className).toContain('calc(25%_-_6px)')
  })

  it('merges a caller className onto the row', () => {
    const { container } = render(<StatTileGrid tiles={tiles} className="mt-2" />)
    const row = container.firstChild as HTMLElement
    expect(row.className).toContain('mt-2')
    expect(row.className).toContain('gap-2')
  })
})
