import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockPath: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    path: mocks.mockPath,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconSearch: () => <span />,

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

import { TokenTreePathCard } from './TokenTreePathCard'

const TRACE = {
  steps: [
    { remaining: 'the</w>', token: 'the</w>', id: 3, consumed: 7 },
    { remaining: ' quick</w>', token: ' quick</w>', id: 27, consumed: 10 },
  ],
  ids: [3, 27],
}

describe('TokenTreePathCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the text input with a default', () => {
    render(<TokenTreePathCard />)
    expect(screen.getByLabelText('Text to trace')).toBeDefined()
    expect((screen.getByLabelText('Text to trace') as HTMLInputElement).value).toBe('the quick brown fox')
  })

  it('traces a path and renders steps with tokens and ids', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)

    fireEvent.change(screen.getByLabelText('Text to trace'), { target: { value: 'the quick' } })
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(mocks.mockPath).toHaveBeenCalledWith('the quick'))

    expect(screen.getByText('#1')).toBeDefined()
    expect(screen.getByText('the</w>')).toBeDefined()
    expect(screen.getByText('id 3')).toBeDefined()
    expect(screen.getByText('[3, 27]')).toBeDefined()
  })

  it('disables the button for a blank text', () => {
    render(<TokenTreePathCard />)
    fireEvent.change(screen.getByLabelText('Text to trace'), { target: { value: '   ' } })
    expect((screen.getByRole('button', { name: /^Trace$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('traces on Enter key', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.keyDown(screen.getByLabelText('Text to trace'), { key: 'Enter' })
    await waitFor(() => expect(mocks.mockPath).toHaveBeenCalledWith('the quick brown fox'))
  })

  it('shows a toast when tracing fails', async () => {
    mocks.mockPath.mockRejectedValue(new Error('boom'))
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Could not trace the token path', 'error'))
  })

  it('renders step numbers as #1, #2, etc.', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(screen.getByText('#1')).toBeDefined())
    expect(screen.getByText('#2')).toBeDefined()
  })

  it('shows consumed character counts', async () => {
    mocks.mockPath.mockResolvedValue(TRACE)
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    await waitFor(() => expect(screen.getByText('+7')).toBeDefined())
    expect(screen.getByText('+10')).toBeDefined()
  })

  it('shows tracing... while loading', async () => {
    let resolvePromise: any
    mocks.mockPath.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreePathCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Trace$/ }))
    expect(screen.getByText('Tracing...')).toBeDefined()
    resolvePromise(TRACE)
    await waitFor(() => expect(screen.getByText('Trace')).toBeDefined())
  })

  it('renders the CardTitle', () => {
    render(<TokenTreePathCard />)
    expect(screen.getByText('Token Path Explorer')).toBeDefined()
  })
})
