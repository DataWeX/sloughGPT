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

import { ExperimentListCard } from './ExperimentListCard'

afterEach(() => cleanup())

const mockExps = [
  { id: 'exp-1', name: 'LR Test', runs: 5, status: 'running' },
  { id: 'exp-2', name: 'Batch Size', runs: 3, status: 'completed' },
  { id: 'exp-3', name: 'Dropout Sweep', status: 'queued' },
]

describe('ExperimentListCard', () => {
  it('renders empty state', () => {
    render(<ExperimentListCard experiments={[]} />)
    expect(screen.getByText('Experiments')).toBeTruthy()
    expect(screen.getByText('(0)')).toBeTruthy()
    expect(screen.getByText('No experiments yet.')).toBeTruthy()
  })

  it('shows experiment list', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('LR Test')).toBeTruthy()
    expect(screen.getByText('Batch Size')).toBeTruthy()
  })

  it('shows status badges', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('queued')).toBeTruthy()
  })

  it('shows run counts', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    expect(screen.getByText('5 runs')).toBeTruthy()
    expect(screen.getByText('3 runs')).toBeTruthy()
  })

  it('filters by search', () => {
    render(<ExperimentListCard experiments={mockExps} />)
    fireEvent.change(screen.getByTestId('experiment-search'), { target: { value: 'LR' } })
    expect(screen.getByText('LR Test')).toBeTruthy()
    expect(screen.queryByText('Batch Size')).toBeNull()
  })

  it('calls onSelect', () => {
    const onSelect = vi.fn()
    render(<ExperimentListCard experiments={mockExps} onSelect={onSelect} />)
    const exp1 = screen.getByTestId('experiment-exp-1')
    const btn = exp1.querySelector('button')
    if (btn) fireEvent.click(btn)
    expect(onSelect).toHaveBeenCalledWith('exp-1')
  })

  it('calls onDelete', () => {
    const onDelete = vi.fn()
    render(<ExperimentListCard experiments={mockExps} onDelete={onDelete} />)
    const delBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Del')
    fireEvent.click(delBtns[0])
    expect(onDelete).toHaveBeenCalledWith('exp-1')
  })

  it('highlights selected experiment', () => {
    render(<ExperimentListCard experiments={mockExps} selectedId="exp-2" />)
    const exp2 = screen.getByTestId('experiment-exp-2')
    expect(exp2.className).toContain('border-primary/50')
  })
})
