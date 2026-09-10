/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { NotificationStatCards } from './NotificationStatCards'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  KpiGrid: ({ children, ...props }: any) => <div data-testid="kpi-grid" {...props}>{children}</div>,
  StatCard: ({ label, value }: any) => (
    <div data-testid="stat-card">
      <span>{label}</span>
      <span>{value}</span>
    </div>
  ),
}))

afterEach(() => cleanup())

describe('NotificationStatCards', () => {
  it('renders three stat cards', () => {
    render(<NotificationStatCards total={10} training={3} members={7} />)
    const cards = screen.getAllByTestId('stat-card')
    expect(cards).toHaveLength(3)
  })

  it('renders total count', () => {
    render(<NotificationStatCards total={42} training={0} members={0} />)
    expect(screen.getByText('Total')).toBeDefined()
    expect(screen.getByText('42')).toBeDefined()
  })

  it('renders training count', () => {
    render(<NotificationStatCards total={0} training={15} members={0} />)
    expect(screen.getByText('Training')).toBeDefined()
    expect(screen.getByText('15')).toBeDefined()
  })

  it('renders members count', () => {
    render(<NotificationStatCards total={0} training={0} members={8} />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('8')).toBeDefined()
  })

  it('renders zero values correctly', () => {
    render(<NotificationStatCards total={0} training={0} members={0} />)
    const cards = screen.getAllByTestId('stat-card')
    expect(cards).toHaveLength(3)
  })
})
