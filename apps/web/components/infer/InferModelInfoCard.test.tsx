// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
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

import { InferModelInfoCard } from './InferModelInfoCard'

afterEach(() => cleanup())

const defaultProps = {
  info: null,
  loading: false,
  onLoad: vi.fn(),
}

describe('InferModelInfoCard', () => {
  it('renders title', () => {
    render(<InferModelInfoCard {...defaultProps} />)
    expect(screen.getByText('Model Information')).toBeTruthy()
  })

  it('renders load button', () => {
    render(<InferModelInfoCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: /Load Info/ })).toBeTruthy()
  })

  it('disables button when loading', () => {
    render(<InferModelInfoCard {...defaultProps} loading={true} />)
    const button = screen.getByRole('button', { name: /Loading/ })
    expect(button.hasAttribute('disabled')).toBe(true)
  })

  it('shows stat boxes when info provided', () => {
    const info = {
      model_id: 'gpt-2',
      model_type: 'causal-lm',
      num_parameters: 124000000,
      vocab_size: 50257,
      max_context: 1024,
      num_layers: 12,
      has_tokenizer: true,
      has_streaming: true,
      has_embedding: false,
    }
    render(<InferModelInfoCard {...defaultProps} info={info} />)
    expect(screen.getByText('gpt-2')).toBeTruthy()
    expect(screen.getByText('causal-lm')).toBeTruthy()
    expect(screen.getByText('124,000,000')).toBeTruthy()
    expect(screen.getByText('50,257')).toBeTruthy()
    expect(screen.getByText('1,024')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
  })

  it('displays boolean stats as Yes/No', () => {
    const info = {
      model_id: 'm',
      model_type: 't',
      num_parameters: 1,
      vocab_size: 1,
      max_context: 1,
      num_layers: 1,
      has_tokenizer: true,
      has_streaming: false,
      has_embedding: true,
    }
    render(<InferModelInfoCard {...defaultProps} info={info} />)
    const yesNos = screen.getAllByText(/^(Yes|No)$/)
    expect(yesNos.length).toBe(3)
  })
})
