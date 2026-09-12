import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: any) => <div data-testid="card">{children}</div>,
  CardContent: ({ children }: any) => <div data-testid="card-content">{children}</div>,
  CardHeader: ({ children }: any) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: any) => <div data-testid="card-title">{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} data-testid="button">{children}</button>
  ),
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label">{children}</label>,

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

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

const { mockApiGet, mockApiPost, mockApiDelete } = vi.hoisted(() => ({
  mockApiGet: vi.fn(),
  mockApiPost: vi.fn(),
  mockApiDelete: vi.fn(),
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: any[]) => mockApiGet(...args),
  apiPost: (...args: any[]) => mockApiPost(...args),
  apiDelete: (...args: any[]) => mockApiDelete(...args),
}))

import CollectionsContent from './collections-content'

beforeEach(() => {
  mockApiGet.mockReset()
  mockApiGet.mockResolvedValue({ pipelines: [], counts: null })
})

afterEach(() => {
  cleanup()
})

describe('CollectionsContent', () => {
  it('renders without crashing', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/Pipelines/)).toBeDefined()
    })
  })

  it('shows loading skeleton initially', () => {
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<CollectionsContent />)
    const skeletons = screen.getAllByTestId('card')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('shows empty state when no pipelines', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/No pipelines configured/)).toBeDefined()
    })
  })

  it('renders pipeline list from API', async () => {
    mockApiGet.mockResolvedValue({
      pipelines: [
        { id: '1', name: 'test-pipe', source_type: 'file', store_type: 'memory', records_count: 10 },
      ],
      counts: { pipelines: 1, sources: 1, stores: 1, filters: 0 },
    })
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText('test-pipe')).toBeDefined()
    })
    expect(screen.getByText('Source: file')).toBeDefined()
    expect(screen.getByText('Store: memory')).toBeDefined()
  })

  it('renders stats cards when counts are provided', async () => {
    mockApiGet.mockResolvedValue({
      pipelines: [],
      counts: { pipelines: 10, sources: 20, stores: 30, filters: 40 },
    })
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText('10')).toBeDefined()
    })
    expect(screen.getByText('20')).toBeDefined()
    expect(screen.getByText('30')).toBeDefined()
    expect(screen.getByText('40')).toBeDefined()
  })

  it('toggles create form on New pipeline click', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(screen.getByText(/No pipelines configured/)).toBeDefined()
    })
    const buttons = screen.getAllByTestId('button')
    const newBtn = buttons.find(b => b.textContent === 'New pipeline')
    expect(newBtn).toBeDefined()
    fireEvent.click(newBtn!)
    expect(screen.getByText('Create pipeline')).toBeDefined()
  })

  it('calls apiGet on mount', async () => {
    render(<CollectionsContent />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/collections')
    })
  })
})
