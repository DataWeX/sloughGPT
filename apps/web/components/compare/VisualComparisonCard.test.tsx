import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  RadarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="radar-chart">{children}</div>,
  Bar: () => null,
  Radar: () => null,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  Legend: () => null,
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

import VisualComparisonCard from '@/components/compare/VisualComparisonCard'

describe('VisualComparisonCard', () => {
  it('returns null when < 2 data points', () => {
    const { container } = render(<VisualComparisonCard chartData={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null with single data point', () => {
    const { container } = render(
      <VisualComparisonCard chartData={[{ name: 'A', throughput: 10, latency: 50, memory: 100 }]} />,
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders when >= 2 data points', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByText('Visual Comparison').length).toBeGreaterThanOrEqual(1)
  })

  it('renders throughput and latency chart labels', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'X', throughput: 5, latency: 100, memory: 200 },
        { name: 'Y', throughput: 8, latency: 80, memory: 150 },
      ]} />,
    )
    expect(screen.getAllByText(/Throughput/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/latency/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders bar charts', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThanOrEqual(2)
  })

  it('passes chartData with name field to recharts', () => {
    const data = [
      { name: 'GPT-2', throughput: 10, latency: 50, memory: 100 },
      { name: 'Qwen', throughput: 20, latency: 30, memory: 80 },
    ]
    render(<VisualComparisonCard chartData={data} />)
    expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThanOrEqual(1)
  })

  it('renders responsive container', () => {
    render(
      <VisualComparisonCard chartData={[
        { name: 'A', throughput: 10, latency: 50, memory: 100 },
        { name: 'B', throughput: 20, latency: 30, memory: 80 },
      ]} />,
    )
    expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThanOrEqual(1)
  })
})
