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
  Textarea: (props: any) => <textarea {...props} />,
  Label: ({ children, ...props }: any) => <label {...props}>{children}</label>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
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

import { InferEmbedCard } from './InferEmbedCard'

afterEach(() => cleanup())

const defaultProps = {
  prompt: '',
  result: null,
  loading: false,
  onPromptChange: vi.fn(),
  onRun: vi.fn(),
}

describe('InferEmbedCard', () => {
  it('renders title', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByText('Text Embedding')).toBeTruthy()
  })

  it('renders prompt textarea', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByLabelText('Embed prompt')).toBeTruthy()
  })

  it('renders run button', () => {
    render(<InferEmbedCard {...defaultProps} />)
    expect(screen.getByRole('button', { name: /Run/ })).toBeTruthy()
  })

  it('shows dimensions when result provided', () => {
    const result = {
      embedding: Array(384).fill(0.1),
      dimensions: 384,
      model: 'embed-model',
    }
    render(<InferEmbedCard {...defaultProps} result={result} />)
    expect(screen.getByText('384 dimensions')).toBeTruthy()
    expect(screen.getByText('embed-model')).toBeTruthy()
  })

  it('renders bar chart when result provided', () => {
    const result = {
      embedding: [0.5, -0.3, 0.8, 0.1, -0.6, 0.2, 0.4, -0.1, 0.7, 0.3, -0.4, 0.6, 0.2, -0.5, 0.9, 0.1, -0.2, 0.3, 0.5, -0.7, 0.4],
      dimensions: 384,
      model: 'embed-model',
    }
    render(<InferEmbedCard {...defaultProps} result={result} />)
    const chart = screen.getByLabelText('Embedding visualization')
    expect(chart.children.length).toBe(20)
  })

  it('disables run button when loading', () => {
    render(<InferEmbedCard {...defaultProps} loading={true} />)
    const button = screen.getByRole('button', { name: /Embedding/ })
    expect(button.hasAttribute('disabled')).toBe(true)
  })
})
