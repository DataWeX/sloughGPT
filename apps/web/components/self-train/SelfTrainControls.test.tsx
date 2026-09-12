// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button data-testid="button" {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),

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

import { SelfTrainControls } from './SelfTrainControls'

const defaultProps = {
  model: '',
  temperature: 0.7,
  forever: false,
  isRunning: false,
  onModelChange: vi.fn(),
  onTemperatureChange: vi.fn(),
  onForeverToggle: vi.fn(),
  onStart: vi.fn(),
  onStop: vi.fn(),
}

describe('SelfTrainControls', () => {
  afterEach(() => cleanup())

  it('renders title "Controls"', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Controls')).toBeTruthy()
  })

  it('renders model and temperature inputs', () => {
    render(<SelfTrainControls {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs.length).toBe(2)
  })

  it('renders start button when not running', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Start self-training')).toBeTruthy()
  })

  it('renders stop button when running', () => {
    render(<SelfTrainControls {...defaultProps} isRunning />)
    expect(screen.getByText('Stop')).toBeTruthy()
  })

  it('shows "Starting..." when starting prop is true', () => {
    render(<SelfTrainControls {...defaultProps} starting />)
    expect(screen.getByText('Starting...')).toBeTruthy()
  })

  it('displays single pass label when forever is false', () => {
    render(<SelfTrainControls {...defaultProps} />)
    expect(screen.getByText('Single pass')).toBeTruthy()
  })

  it('displays train forever label when forever is true', () => {
    render(<SelfTrainControls {...defaultProps} forever />)
    expect(screen.getByText('Train forever')).toBeTruthy()
  })
})
