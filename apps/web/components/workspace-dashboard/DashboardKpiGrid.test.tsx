/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { DashboardKpiGrid } from './DashboardKpiGrid'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  StatCard: ({ label, value, ...props }: any) => <div data-testid="stat-card" {...props}><span>{label}</span>: <span>{String(value)}</span></div>,
  KpiGrid: ({ children, ...props }: any) => <div data-testid="kpi-grid" {...props}>{children}</div>,
}))

describe('DashboardKpiGrid', () => {
  const defaultProps = {
    kpis: [
      { label: 'Members', value: 5 },
      { label: 'Datasets', value: 12 },
      { label: 'API Keys', value: 3 },
    ],
  }

  it('renders the kpi grid', () => {
    render(<DashboardKpiGrid {...defaultProps} />)
    expect(screen.getByTestId('kpi-grid')).toBeDefined()
  })

  it('renders all kpi items', () => {
    render(<DashboardKpiGrid {...defaultProps} />)
    expect(screen.getAllByTestId('stat-card')).toHaveLength(3)
  })

  it('renders kpi labels', () => {
    render(<DashboardKpiGrid {...defaultProps} />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('Datasets')).toBeDefined()
    expect(screen.getByText('API Keys')).toBeDefined()
  })

  it('renders kpi values', () => {
    render(<DashboardKpiGrid {...defaultProps} />)
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('12')).toBeDefined()
    expect(screen.getByText('3')).toBeDefined()
  })

  it('renders empty grid when no kpis', () => {
    render(<DashboardKpiGrid kpis={[]} />)
    expect(screen.getByTestId('kpi-grid')).toBeDefined()
    expect(screen.queryAllByTestId('stat-card')).toHaveLength(0)
  })

  it('renders string values', () => {
    render(<DashboardKpiGrid kpis={[{ label: 'Name', value: 'My Workspace' }]} />)
    expect(screen.getByText('My Workspace')).toBeDefined()
  })
})
