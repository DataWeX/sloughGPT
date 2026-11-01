import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

const mocks = vi.hoisted(() => ({
  mockLineage: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    lineage: mocks.mockLineage,
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
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  Skeleton: ({ className }: { className?: string }) => (
    <div data-testid="skeleton" className={className} />
  ),
  IconActivity: () => <span data-testid="icon-activity" />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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

import { TokenTreeLineageCard } from './TokenTreeLineageCard'

const LINEAGE = {
  token: 'quick</w>',
  leaves: ['q', 'u', 'i', 'c', 'k', '</w>'],
  tree: 'quick</w>\n  qu\n    ick',
}

describe('TokenTreeLineageCard', () => {
  beforeEach(() => {
    mocks.mockLineage.mockResolvedValue(LINEAGE)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('fetches and renders a token lineage on click', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('quick'))
    expect(await screen.findByText(/Merge lineage of/)).toBeDefined()
    expect(screen.getByText('q')).toBeDefined()
    expect(screen.getByText('</w>')).toBeDefined()
    expect(screen.getByText(/quick<\/w>/)).toBeDefined()
  })

  it('shows a skeleton while loading', async () => {
    mocks.mockLineage.mockImplementation(() => new Promise(() => {}))
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    expect(await screen.findByTestId('skeleton')).toBeDefined()
  })

  it('triggers lookup on Enter key', async () => {
    render(<TokenTreeLineageCard />)
    const input = screen.getByLabelText('Token to inspect')
    fireEvent.change(input, { target: { value: 'fox' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() => expect(mocks.mockLineage).toHaveBeenCalledWith('fox'))
  })

  it('shows a toast when the token is not in the vocabulary', async () => {
    mocks.mockLineage.mockRejectedValue(new Error('not found'))
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'zzz' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('Token not in vocabulary: zzz', 'error'))
    expect(screen.queryByText(/Merge lineage of/)).toBeNull()
  })

  it('disables the button when the input is empty', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: '   ' } })
    const button = screen.getByText('Show lineage').closest('button')
    expect(button).not.toBeNull()
    expect(button?.disabled).toBe(true)
    expect(mocks.mockLineage).not.toHaveBeenCalled()
  })

  it('displays the merge tree as preformatted text', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => {
      expect(screen.getByText('q')).toBeDefined()
    })
    expect(document.querySelector('pre')?.textContent).toContain('quick')
  })

  it('shows character leaf count', async () => {
    render(<TokenTreeLineageCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'quick' } })
    fireEvent.click(screen.getByText('Show lineage'))
    await waitFor(() => {
      expect(screen.getByText('q')).toBeDefined()
    })
    expect(screen.getByText(/6 character leaves/)).toBeDefined()
  })

  it('renders the CardTitle', () => {
    render(<TokenTreeLineageCard />)
    expect(screen.getByText('Merge Lineage Explorer')).toBeDefined()
  })
})
