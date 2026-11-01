// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn((key: string, value: unknown) => {
      localStorage.setItem(key, JSON.stringify(value))
      return Promise.resolve()
    }),
  },
}))

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

import { ExportScheduleCard } from './ExportScheduleCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ExportScheduleCard', () => {
  it('renders empty state', async () => {
    render(<ExportScheduleCard />)
    await waitFor(() => {
      expect(screen.getByText('Scheduled Exports')).toBeTruthy()
    })
    expect(screen.getByText('No scheduled exports.')).toBeTruthy()
  })

  it('opens new schedule form', async () => {
    render(<ExportScheduleCard />)
    await waitFor(() => { expect(screen.getByText('Scheduled Exports')).toBeTruthy() })
    fireEvent.click(screen.getByText('+ New'))
    expect(screen.getByPlaceholderText('Schedule name')).toBeTruthy()
    expect(screen.getByText('Save Schedule')).toBeTruthy()
  })

  it('saves a schedule', async () => {
    render(<ExportScheduleCard />)
    await waitFor(() => { expect(screen.getByText('Scheduled Exports')).toBeTruthy() })
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Schedule name'), { target: { value: 'Weekly Backup' } })
    fireEvent.click(screen.getByText('Save Schedule'))
    await waitFor(() => {
      expect(screen.getByText('Weekly Backup')).toBeTruthy()
    })
  })

  it('loads schedules from localStorage', async () => {
    const nextRun = Date.now() + 86400000
    localStorage.setItem('sloughgpt-export-schedules', JSON.stringify([
      { id: 's1', name: 'Daily Model', type: 'model', format: 'sou', frequency: 'daily', enabled: true, nextRun, timestamp: 1 },
    ]))
    render(<ExportScheduleCard />)
    await waitFor(() => {
      expect(screen.getByText('Daily Model')).toBeTruthy()
    })
  })

  it('toggles schedule enabled/disabled', async () => {
    const nextRun = Date.now() + 86400000
    localStorage.setItem('sloughgpt-export-schedules', JSON.stringify([
      { id: 's1', name: 'Test', type: 'model', format: 'sou', frequency: 'daily', enabled: true, nextRun, timestamp: 1 },
    ]))
    render(<ExportScheduleCard />)
    await waitFor(() => { expect(screen.getByText('Test')).toBeTruthy() })
    const buttons = screen.getAllByRole('button')
    const pauseBtn = buttons.find(b => b.textContent?.includes('Pause'))
    if (pauseBtn) fireEvent.click(pauseBtn)
    await waitFor(() => {
      expect(screen.getByText('Resume')).toBeTruthy()
    })
  })

  it('deletes a schedule', async () => {
    const nextRun = Date.now() + 86400000
    localStorage.setItem('sloughgpt-export-schedules', JSON.stringify([
      { id: 's1', name: 'To Delete', type: 'model', format: 'sou', frequency: 'daily', enabled: true, nextRun, timestamp: 1 },
    ]))
    render(<ExportScheduleCard />)
    await waitFor(() => { expect(screen.getByText('To Delete')).toBeTruthy() })
    const buttons = screen.getAllByRole('button')
    const delBtn = buttons.find(b => b.textContent?.includes('Del'))
    if (delBtn) fireEvent.click(delBtn)
    await waitFor(() => {
      expect(screen.queryByText('To Delete')).toBeNull()
    })
  })
})
