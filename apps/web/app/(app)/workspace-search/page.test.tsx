import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const mockApiGet = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => '/workspace-search',
}))

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.mock('@/lib/auth', () => ({
  useAuthStore: (selector?: (s: { currentWorkspace: { id: string } | null }) => unknown) => {
    const state = { currentWorkspace: { id: 'ws-1' } }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconSearch: (props: Record<string, unknown>) => <svg data-testid="icon-search" {...props} />,
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title }: any) => (
    <div data-testid="page-container" data-title={title}><h1>{title}</h1>{children}</div>
  ),
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ children }: any) => <div>{children}</div>,
  AppRouteHeaderLead: ({ children }: any) => <div>{children}</div>,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
    Input: ({ value, onChange, placeholder, ...props }: any) => <input value={value} onChange={onChange} placeholder={placeholder} {...props} />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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
}
})

vi.mock('lucide-react', () => ({
  Users: () => <span data-testid="icon-users" />,
  Brain: () => <span data-testid="icon-brain" />,
  Database: () => <span data-testid="icon-database" />,
  BookOpen: () => <span data-testid="icon-book" />,
  ExternalLink: () => <span data-testid="icon-external" />,
}))

import WorkspaceSearchPage from './page'

describe('WorkspaceSearchPage', () => {
  const mockSearchResults = {
    data: {
      results: {
        members: [
          { id: 'u1', type: 'member', title: 'alice', detail: 'alice@test.com' },
        ],
        training_jobs: [
          { id: 't1', type: 'training', title: 'Fine-tune LLaMA', detail: 'completed' },
        ],
        datasets: [
          { id: 'd1', type: 'dataset', title: 'training-data', detail: '1000 rows' },
        ],
        knowledge: [],
      },
      total: 3,
    },
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockApiGet.mockResolvedValue(mockSearchResults)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders page header', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getAllByText('Workspace Search').length).toBeGreaterThanOrEqual(1)
  })

  it('shows search input', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getByPlaceholderText(/search/i)).toBeTruthy()
  })

  it('shows initial prompt', async () => {
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')
    expect(screen.getAllByText(/type to search/i).length).toBeGreaterThanOrEqual(1)
  })

  it('performs search on input', async () => {
    const user = userEvent.setup()
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')

    const searchInput = screen.getByPlaceholderText(/search/i)
    await user.type(searchInput, 'alice')

    await screen.findByText(/result/i)
    expect(mockApiGet).toHaveBeenCalled()
  })

  it('shows empty state when no results', async () => {
    mockApiGet.mockResolvedValue({
      data: { results: { members: [], training_jobs: [], datasets: [], knowledge: [] }, total: 0 },
    })
    const user = userEvent.setup()
    render(<WorkspaceSearchPage />)
    await screen.findAllByText('Workspace Search')

    const searchInput = screen.getByPlaceholderText(/search/i)
    await user.type(searchInput, 'zzznonexistent')

    await screen.findByText(/no results/i)
  })
})
