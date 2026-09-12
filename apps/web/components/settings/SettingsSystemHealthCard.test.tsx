// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Skeleton: ({ ...props }: any) => <span {...props} />,
  StatCard: ({ label, value }: any) => <div>{label}: {value}</div>,
  KpiGrid: ({ children }: any) => <div>{children}</div>,
  IconRefresh: ({ className }: any) => <span className={className}>refresh</span>,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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

vi.mock('@/lib/chat-utils', () => ({
  formatUptime: (s: number) => `${s}s`,
}))

import { SettingsSystemHealthCard } from './SettingsSystemHealthCard'

afterEach(() => cleanup())

const defaultProps = {
  apiOk: true,
  modelLoaded: true,
  modelType: 'llama',
  detailed: { status: 'healthy', uptime_seconds: 3600, inference: { inference_count: 42 }, gpu: { backend: 'cuda', tier: 'high' }, versions: { package: '1.0.0', api: '2.0.0' } },
  metrics: { cpu_percent: 25, memory_used_gb: 4.0, memory_total_gb: 16.0, memory_percent: 25 },
  disk: { used_gb: 100, total_gb: 500, percent: 20 },
  info: { platform: 'linux', platform_release: '6.1', architecture: 'x86_64', processor: 'Intel', cpu_count: 8, platform_version: '6.1.0' },
  healthError: false,
  onRefresh: vi.fn(),
}

describe('SettingsSystemHealthCard', () => {
  it('renders title and description', () => {
    render(<SettingsSystemHealthCard {...defaultProps} />)
    expect(screen.getByText('System health')).toBeTruthy()
    expect(screen.getByText('Backend status and resource usage')).toBeTruthy()
  })

  it('displays health error state', () => {
    render(<SettingsSystemHealthCard {...defaultProps} healthError={true} detailed={null} metrics={null} disk={null} info={null} />)
    expect(screen.getByText('Could not connect to service')).toBeTruthy()
  })

  it('renders retry button on error', () => {
    render(<SettingsSystemHealthCard {...defaultProps} healthError={true} detailed={null} metrics={null} disk={null} info={null} />)
    expect(screen.getByText('Retry')).toBeTruthy()
  })

  it('displays API healthy status', () => {
    render(<SettingsSystemHealthCard {...defaultProps} />)
    expect(screen.getByText('Healthy')).toBeTruthy()
  })

  it('displays model type', () => {
    render(<SettingsSystemHealthCard {...defaultProps} />)
    expect(screen.getByText('llama')).toBeTruthy()
  })

  it('displays platform info', () => {
    render(<SettingsSystemHealthCard {...defaultProps} />)
    expect(screen.getByText(/linux 6.1/)).toBeTruthy()
  })

  it('displays GPU info', () => {
    render(<SettingsSystemHealthCard {...defaultProps} />)
    expect(screen.getByText(/CUDA · high/)).toBeTruthy()
  })
})
