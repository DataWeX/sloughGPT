// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Skeleton: (props: any) => <div data-testid="skeleton" {...props} />,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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

import { CloudTrainingJobList } from './CloudTrainingJobList'
import type { CloudJob } from './CloudTrainingJobList'

afterEach(() => cleanup())

const mockJobs: CloudJob[] = [
  { job_id: 'job-1', provider: 'aws', status: 'completed', progress: 100 },
  { job_id: 'job-2', provider: 'gcp', status: 'running', progress: 50 },
  { job_id: 'job-3', provider: 'local', status: 'failed', progress: 20, error: 'OOM' },
]

describe('CloudTrainingJobList', () => {
  it('renders the card title', () => {
    render(<CloudTrainingJobList />)
    expect(screen.getAllByText('Training Jobs').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no jobs', () => {
    render(<CloudTrainingJobList />)
    expect(screen.getByText('No training jobs yet.')).toBeTruthy()
  })

  it('shows skeleton when loading', () => {
    render(<CloudTrainingJobList loading />)
    expect(screen.getByTestId('skeleton')).toBeTruthy()
  })

  it('renders jobs when provided', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('job-1')).toBeTruthy()
    expect(screen.getByText('job-2')).toBeTruthy()
    expect(screen.getByText('job-3')).toBeTruthy()
  })

  it('renders correct number of badges', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    const badges = screen.getAllByText(/completed|running|failed/)
    expect(badges.length).toBe(3)
  })

  it('renders correct status text in badges', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByText('failed')).toBeTruthy()
  })

  it('renders provider names', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('aws')).toBeTruthy()
    expect(screen.getByText('gcp')).toBeTruthy()
    expect(screen.getByText('local')).toBeTruthy()
  })
})
