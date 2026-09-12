/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { WorldTickResultCard } from './WorldTickResultCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

describe('WorldTickResultCard', () => {
  it('renders nothing when both results are null', () => {
    const { container } = render(<WorldTickResultCard tickResult={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders tick result title', () => {
    render(<WorldTickResultCard tickResult={{ tick: 1, babies: 3 }} />)
    expect(screen.getByText('Tick Result')).toBeDefined()
  })

  it('renders tick number', () => {
    render(<WorldTickResultCard tickResult={{ tick: 5, babies: 0 }} />)
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders babies count', () => {
    render(<WorldTickResultCard tickResult={{ tick: 1, babies: 12 }} />)
    expect(screen.getByText('12')).toBeDefined()
  })

  it('renders neural processing title when neuralResult provided', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ embedding_shape: [128, 64] }}
      />,
    )
    expect(screen.getByText('Neural Processing')).toBeDefined()
  })

  it('renders embedding shape', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ embedding_shape: [128, 64] }}
      />,
    )
    expect(screen.getByText('128,64')).toBeDefined()
  })

  it('renders N/A when no embedding shape', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ descriptor: { foo: 'bar' } }}
      />,
    )
    expect(screen.getByText('N/A')).toBeDefined()
  })

  it('renders descriptor JSON', () => {
    render(
      <WorldTickResultCard
        tickResult={{ tick: 1, babies: 0 }}
        neuralResult={{ descriptor: { key: 'value' } }}
      />,
    )
    expect(screen.getByText(/"key": "value"/)).toBeDefined()
  })

  it('renders neural card even without tick result', () => {
    render(
      <WorldTickResultCard
        tickResult={null}
        neuralResult={{ embedding_shape: [10] }}
      />,
    )
    expect(screen.getByText('Neural Processing')).toBeDefined()
  })
})
