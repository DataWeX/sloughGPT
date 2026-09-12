// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
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

import { AutoTrainConfigCard } from './AutoTrainConfigCard'

afterEach(() => cleanup())

describe('AutoTrainConfigCard', () => {
  it('renders config form', () => {
    render(<AutoTrainConfigCard onSave={vi.fn()} />)
    expect(screen.getByText('Configuration')).toBeTruthy()
    expect(screen.getByText('Pair Threshold')).toBeTruthy()
    expect(screen.getByText('Check Interval (seconds)')).toBeTruthy()
    expect(screen.getByText('Save Configuration')).toBeTruthy()
  })

  it('shows default values', () => {
    render(<AutoTrainConfigCard onSave={vi.fn()} />)
    expect(screen.getByTestId('threshold-input')).toHaveValue(10)
    expect(screen.getByTestId('interval-input')).toHaveValue(120)
  })

  it('shows custom values', () => {
    render(<AutoTrainConfigCard threshold={50} intervalS={300} onSave={vi.fn()} />)
    expect(screen.getByTestId('threshold-input')).toHaveValue(50)
    expect(screen.getByTestId('interval-input')).toHaveValue(300)
  })

  it('updates threshold', () => {
    render(<AutoTrainConfigCard />)
    fireEvent.change(screen.getByTestId('threshold-input'), { target: { value: '25' } })
    expect(screen.getByTestId('threshold-input')).toHaveValue(25)
  })

  it('calls onSave', async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    render(<AutoTrainConfigCard onSave={onSave} />)
    fireEvent.change(screen.getByTestId('threshold-input'), { target: { value: '30' } })
    fireEvent.click(screen.getByText('Save Configuration'))
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(30, 120)
    })
  })

  it('shows saving state', async () => {
    const onSave = vi.fn().mockImplementation(() => new Promise(r => setTimeout(r, 100)))
    render(<AutoTrainConfigCard onSave={onSave} />)
    fireEvent.click(screen.getByText('Save Configuration'))
    expect(screen.getByText('Saving...')).toBeTruthy()
  })
})
