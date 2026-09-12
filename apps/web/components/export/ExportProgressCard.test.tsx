// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,

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

import { ExportProgressCard, startExportJob, updateExportJob } from './ExportProgressCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ExportProgressCard', () => {
  it('renders nothing when no jobs', () => {
    const { container } = render(<ExportProgressCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows running jobs', () => {
    const id = startExportJob('Model Export')
    updateExportJob(id, { progress: 50, message: 'Processing...' })
    render(<ExportProgressCard />)
    expect(screen.getByText('Export Progress')).toBeTruthy()
    expect(screen.getByText('Model Export')).toBeTruthy()
    expect(screen.getByText('50%')).toBeTruthy()
    expect(screen.getByText('Processing...')).toBeTruthy()
  })

  it('shows completed jobs', () => {
    const id = startExportJob('Training Data')
    updateExportJob(id, { status: 'completed', progress: 100 })
    render(<ExportProgressCard />)
    expect(screen.getByText('completed')).toBeTruthy()
  })

  it('shows failed jobs with error', () => {
    const id = startExportJob('Checkpoint')
    updateExportJob(id, { status: 'failed', error: 'Disk full' })
    render(<ExportProgressCard />)
    expect(screen.getByText('failed')).toBeTruthy()
    expect(screen.getByText('Disk full')).toBeTruthy()
  })

  it('shows active count badge', () => {
    startExportJob('Job 1')
    startExportJob('Job 2')
    render(<ExportProgressCard />)
    expect(screen.getByText('2 active')).toBeTruthy()
  })

  it('clears completed jobs', () => {
    const id = startExportJob('Done Job')
    updateExportJob(id, { status: 'completed', progress: 100 })
    render(<ExportProgressCard />)
    fireEvent.click(screen.getByText('Clear done'))
    expect(screen.queryByText('Done Job')).toBeNull()
  })
})
