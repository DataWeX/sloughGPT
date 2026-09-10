/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { UsageKpiGrid, type KpiItem } from './UsageKpiGrid'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const items: KpiItem[] = [
  { label: 'Members', value: 12 },
  { label: 'Datasets', value: 5 },
  { label: 'API Keys', value: 3 },
  { label: 'Knowledge', value: 8 },
]

describe('UsageKpiGrid', () => {
  it('renders all KPI labels', () => {
    render(<UsageKpiGrid items={items} />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('Datasets')).toBeDefined()
    expect(screen.getByText('API Keys')).toBeDefined()
    expect(screen.getByText('Knowledge')).toBeDefined()
  })

  it('renders KPI values', () => {
    render(<UsageKpiGrid items={items} />)
    expect(screen.getByText('12')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders correct number of cards', () => {
    render(<UsageKpiGrid items={items} />)
    const cards = screen.getAllByTestId('card')
    expect(cards.length).toBe(4)
  })

  it('renders string values', () => {
    render(<UsageKpiGrid items={[{ label: 'Provider', value: 'In Memory' }]} />)
    expect(screen.getByText('In Memory')).toBeDefined()
  })

  it('renders empty grid gracefully', () => {
    render(<UsageKpiGrid items={[]} />)
    const cards = screen.queryAllByTestId('card')
    expect(cards.length).toBe(0)
  })
})
