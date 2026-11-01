/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { UsageKpiGrid, type KpiItem } from './UsageKpiGrid'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,

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

afterEach(() => cleanup())

const items: KpiItem[] = [
  { label: 'Members', value: 12 },
  { label: 'Datasets', value: 5 },
  { label: 'API Keys', value: 3 },
  { label: 'Knowledge', value: 8 },
]

describe('UsageKpiGrid', () => {
  it('renders all KPI labels', () => {
    render(<UsageKpiGrid items={items} />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('Datasets')).toBeDefined()
    expect(screen.getByText('API Keys')).toBeDefined()
    expect(screen.getByText('Knowledge')).toBeDefined()
  })

  it('renders KPI values', () => {
    render(<UsageKpiGrid items={items} />)
    expect(screen.getByText('12')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders correct number of cards', () => {
    render(<UsageKpiGrid items={items} />)
    const cards = screen.getAllByTestId('card')
    expect(cards.length).toBe(4)
  })

  it('renders string values', () => {
    render(<UsageKpiGrid items={[{ label: 'Provider', value: 'In Memory' }]} />)
    expect(screen.getByText('In Memory')).toBeDefined()
  })

  it('renders empty grid gracefully', () => {
    render(<UsageKpiGrid items={[]} />)
    const cards = screen.queryAllByTestId('card')
    expect(cards.length).toBe(0)
  })
})
