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
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: ({ ...props }: any) => <input {...props} />,

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

import { DatasetStatsCard } from './DatasetStatsCard'

afterEach(() => cleanup())

describe('DatasetStatsCard', () => {
  const defaultProps = {
    totalDatasets: 12,
    totalSize: 5242880,
    totalRows: 150000,
    recentUploads: 3,
  }

  it('renders title', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Dataset Overview')).toBeTruthy()
  })

  it('shows total datasets', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Datasets')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
  })

  it('shows total size formatted', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Size')).toBeTruthy()
    expect(screen.getByText('5.0 MB')).toBeTruthy()
  })

  it('shows total rows formatted', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Rows')).toBeTruthy()
    expect(screen.getByText('150,000')).toBeTruthy()
  })

  it('shows recent uploads', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Recent Uploads')).toBeTruthy()
    expect(screen.getByText('3')).toBeTruthy()
  })

  it('renders all four stat boxes', () => {
    render(<DatasetStatsCard {...defaultProps} />)
    expect(screen.getByText('Total Datasets')).toBeTruthy()
    expect(screen.getByText('Total Size')).toBeTruthy()
    expect(screen.getByText('Total Rows')).toBeTruthy()
    expect(screen.getByText('Recent Uploads')).toBeTruthy()
  })
})
