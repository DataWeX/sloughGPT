// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
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
    deleteKV: vi.fn((key: string) => {
      localStorage.removeItem(key)
      return Promise.resolve()
    }),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
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

import { ExportHistoryCard, recordExport } from './ExportHistoryCard'

const STORAGE_KEY = 'sloughgpt-export-history'

afterEach(() => {
  cleanup()
  localStorage.removeItem(STORAGE_KEY)
})

beforeEach(() => {
  localStorage.removeItem(STORAGE_KEY)
})

describe('ExportHistoryCard', () => {
  it('renders empty state for empty history', async () => {
    const { container } = render(<ExportHistoryCard />)
    await waitFor(() => {})
    expect(container.innerHTML).toBe('')
  })

  it('renders when history exists', async () => {
    await recordExport('sou', 3)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByTestId('export-history').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Export History').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total exports', async () => {
    await recordExport('sou', 2)
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows formats used count', async () => {
    await recordExport('sou', 1)
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows total files', async () => {
    await recordExport('sou', 3)
    await recordExport('onnx', 2)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows last export time', async () => {
    await recordExport('sou', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/Last Export/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows recent entries', async () => {
    await recordExport('sou', 1)
    await recordExport('gguf_q4_k_m', 2)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Sou').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Gguf Q4 K M').length).toBeGreaterThanOrEqual(1)
  })

  it('shows file count per entry', async () => {
    await recordExport('sou', 3)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/3 files/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows singular file for count 1', async () => {
    await recordExport('onnx', 1)
    render(<ExportHistoryCard />)
    await waitFor(() => {
      expect(screen.getAllByText(/1 file/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('recordExport persists to localStorage', async () => {
    await recordExport('sou', 2)
    const raw = localStorage.getItem(STORAGE_KEY)
    expect(raw).toBeTruthy()
    const arr = JSON.parse(raw!)
    expect(arr.length).toBe(1)
    expect(arr[0].format).toBe('sou')
    expect(arr[0].fileCount).toBe(2)
  })
})
