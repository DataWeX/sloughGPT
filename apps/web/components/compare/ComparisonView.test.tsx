// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
  IconRefresh: (props: any) => <span data-testid="icon-refresh" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,
}))

vi.mock('next/dynamic', () => {
  const mock = (factory: any, opts?: any) => {
    const Component = React.forwardRef((props: any, ref: any) => React.createElement('div', { 'data-testid': 'dynamic-component', ref }, null))
    return Object.assign(Component, { displayName: 'DynamicComponent' })
  }
  return { __esModule: true, default: mock }
})

vi.mock('@/components/compare/ModelsCard', () => ({
  default: (props: any) => <div data-testid="models-card">{props.models?.length} models</div>,
}))

vi.mock('@/components/compare/ComparisonTableCard', () => ({
  default: (props: any) => <div data-testid="comparison-table-card" />,
}))

vi.mock('@/components/compare/SummaryCard', () => ({
  default: (props: any) => <div data-testid="summary-card" />,
}))

vi.mock('@/components/compare/ModelComparisonInsightsCard', () => ({
  ModelComparisonInsightsCard: (props: any) => <div data-testid="insights-card" />,
}))

vi.mock('@/components/compare/OutputComparisonCard', () => ({
  default: (props: any) => <div data-testid="output-comparison-card" />,
}))

vi.mock('@/components/compare/VisualComparisonCard', () => ({
  default: (props: any) => <div data-testid="visual-comparison-card" />,
}))

import { ComparisonView, ComparisonHeader } from './ComparisonView'

afterEach(() => cleanup())

const baseProps = {
  models: [{ id: 'm1', name: 'Model A' }],
  loading: false,
  results: {},
  running: new Set<string>(),
  snapshots: [],
  snapshotName: '',
  onSnapshotNameChange: vi.fn(),
  completedResults: [],
  bestMetrics: {},
  chartData: [],
  onBenchmark: vi.fn(),
  onClear: vi.fn(),
  onRunAll: vi.fn(),
  onExport: vi.fn(),
  onSaveSnapshot: vi.fn(),
  onLoadSnapshot: vi.fn(),
  onDeleteSnapshot: vi.fn(),
}

describe('ComparisonView', () => {
  it('renders without crashing', () => {
    render(<ComparisonView {...baseProps} />)
    expect(screen.getByText('No benchmark results yet.')).toBeTruthy()
  })

  it('shows empty state message when no results', () => {
    render(<ComparisonView {...baseProps} />)
    expect(screen.getByText('No benchmark results yet.')).toBeTruthy()
    expect(screen.getAllByText('Benchmark all').length).toBeGreaterThanOrEqual(1)
  })

  it('renders models card', () => {
    render(<ComparisonView {...baseProps} />)
    expect(screen.getByTestId('models-card')).toBeTruthy()
    expect(screen.getByText('1 models')).toBeTruthy()
  })

  it('renders snapshots when present', () => {
    const snapshots = [{ id: 's1', name: 'Snapshot 1', savedAt: '2026-01-01T00:00:00Z', modelIds: ['m1'], results: {} }]
    render(<ComparisonView {...baseProps} snapshots={snapshots} />)
    expect(screen.getByText('Saved Comparisons')).toBeTruthy()
    expect(screen.getByText('Snapshot 1')).toBeTruthy()
  })

  it('renders comparison table and cards when results exist', () => {
    const results: [string, any] = [['m1', { throughput: 100, latency: 50 }]]
    render(<ComparisonView {...baseProps} completedResults={results} />)
    expect(screen.getByTestId('comparison-table-card')).toBeTruthy()
    expect(screen.getByTestId('insights-card')).toBeTruthy()
    expect(screen.getByTestId('summary-card')).toBeTruthy()
    expect(screen.getAllByTestId('dynamic-component').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onRunAll when benchmark all clicked', () => {
    const onRunAll = vi.fn()
    render(<ComparisonView {...baseProps} onRunAll={onRunAll} />)
    const btn = screen.getByRole('button', { name: /Benchmark all/ })
    fireEvent.click(btn)
    expect(onRunAll).toHaveBeenCalled()
  })
})

describe('ComparisonHeader', () => {
  it('renders without crashing', () => {
    render(<ComparisonHeader completedResults={[]} snapshotName="" onSnapshotNameChange={vi.fn()} onSaveSnapshot={vi.fn()} onExport={vi.fn()} onRunAll={vi.fn()} loading={false} running={new Set()} />)
  })

  it('shows save input when results exist', () => {
    const results: [string, any] = [['m1', {}]]
    render(<ComparisonHeader completedResults={results} snapshotName="test" onSnapshotNameChange={vi.fn()} onSaveSnapshot={vi.fn()} onExport={vi.fn()} onRunAll={vi.fn()} loading={false} running={new Set()} />)
    expect(screen.getByRole('button', { name: /Save/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Export/ })).toBeTruthy()
  })

  it('hides save input when no results', () => {
    render(<ComparisonHeader completedResults={[]} snapshotName="" onSnapshotNameChange={vi.fn()} onSaveSnapshot={vi.fn()} onExport={vi.fn()} onRunAll={vi.fn()} loading={false} running={new Set()} />)
    expect(screen.queryByRole('button', { name: /Save/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Export/ })).toBeNull()
  })

  it('disables benchmark all when loading', () => {
    render(<ComparisonHeader completedResults={[]} snapshotName="" onSnapshotNameChange={vi.fn()} onSaveSnapshot={vi.fn()} onExport={vi.fn()} onRunAll={vi.fn()} loading={true} running={new Set()} />)
    expect(screen.getByRole('button', { name: /Benchmark all/ })).toHaveProperty('disabled', true)
  })
})
