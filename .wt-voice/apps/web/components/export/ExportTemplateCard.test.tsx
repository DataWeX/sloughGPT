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

import { ExportTemplateCard } from './ExportTemplateCard'

afterEach(() => { cleanup(); localStorage.clear() })

describe('ExportTemplateCard', () => {
  it('renders empty state', async () => {
    render(<ExportTemplateCard />)
    await waitFor(() => {
      expect(screen.getByText('Export Templates')).toBeTruthy()
    })
    expect(screen.getByText('No templates saved yet.')).toBeTruthy()
  })

  it('opens new template form', async () => {
    render(<ExportTemplateCard />)
    await waitFor(() => { expect(screen.getByText('Export Templates')).toBeTruthy() })
    fireEvent.click(screen.getByText('+ New'))
    expect(screen.getByPlaceholderText('Template name')).toBeTruthy()
    expect(screen.getByText('Save Template')).toBeTruthy()
  })

  it('saves a template', async () => {
    render(<ExportTemplateCard />)
    await waitFor(() => { expect(screen.getByText('Export Templates')).toBeTruthy() })
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Template name'), { target: { value: 'My Export' } })
    fireEvent.click(screen.getByText('Save Template'))
    await waitFor(() => {
      expect(screen.getByText('My Export')).toBeTruthy()
    })
  })

  it('loads templates from localStorage', async () => {
    localStorage.setItem('sloughgpt-export-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Saved Template', format: 'onnx', includeTokenizer: false, outputPath: 'out', timestamp: 1 },
    ]))
    render(<ExportTemplateCard />)
    await waitFor(() => {
      expect(screen.getByText('Saved Template')).toBeTruthy()
    })
  })

  it('deletes a template', async () => {
    localStorage.setItem('sloughgpt-export-templates', JSON.stringify([
      { id: 'tpl-1', name: 'To Delete', format: 'sou', includeTokenizer: true, outputPath: 'out', timestamp: 1 },
    ]))
    render(<ExportTemplateCard />)
    await waitFor(() => { expect(screen.getByText('To Delete')).toBeTruthy() })
    const buttons = screen.getAllByRole('button')
    const delBtn = buttons.find(b => b.textContent?.includes('Del'))
    if (delBtn) fireEvent.click(delBtn)
    await waitFor(() => {
      expect(screen.queryByText('To Delete')).toBeNull()
    })
  })

  it('calls onSelect when Use clicked', async () => {
    const onSelect = vi.fn()
    localStorage.setItem('sloughgpt-export-templates', JSON.stringify([
      { id: 'tpl-1', name: 'Test', format: 'sou', includeTokenizer: true, outputPath: 'out', timestamp: 1 },
    ]))
    render(<ExportTemplateCard onSelect={onSelect} />)
    await waitFor(() => { expect(screen.getByText('Test')).toBeTruthy() })
    const buttons = screen.getAllByRole('button')
    const useBtn = buttons.find(b => b.textContent?.includes('Use'))
    if (useBtn) fireEvent.click(useBtn)
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ name: 'Test' }))
  })
})
