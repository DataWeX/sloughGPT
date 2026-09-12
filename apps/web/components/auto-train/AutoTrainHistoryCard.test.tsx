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

import { AutoTrainHistoryCard } from './AutoTrainHistoryCard'

afterEach(() => cleanup())

describe('AutoTrainHistoryCard', () => {
  it('shows empty state', () => {
    render(<AutoTrainHistoryCard lastTrain={null} />)
    expect(screen.getByText('Last Training Run')).toBeTruthy()
    expect(screen.getByText('No training runs yet.')).toBeTruthy()
  })

  it('shows training run details', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: '2026-09-09T10:05:00Z',
      pairs_used: 25,
      checkpoint: 'lora-v3',
    }} />)
    expect(screen.getByText('Started')).toBeTruthy()
    expect(screen.getByText('Completed')).toBeTruthy()
    expect(screen.getByText('25')).toBeTruthy()
    expect(screen.getByText('lora-v3')).toBeTruthy()
  })

  it('shows in-progress state', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: null,
      pairs_used: 10,
      checkpoint: 'lora-v4',
    }} />)
    expect(screen.getByText('In progress...')).toBeTruthy()
  })

  it('shows duration', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: '2026-09-09T10:05:00Z',
      pairs_used: 15,
      checkpoint: 'lora-5',
    }} />)
    expect(screen.getByText(/Duration:/)).toBeTruthy()
  })
})
