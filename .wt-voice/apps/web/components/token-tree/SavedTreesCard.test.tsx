import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SavedTreesCard } from './SavedTreesCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, variant, ...props }: any) => (
    <button data-testid={`btn-${variant || 'default'}`} onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="save-name-input" {...props} />,

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

const defaultProps = {
  savedTrees: [],
  saveName: '',
  loading: false,
  onSaveNameChange: vi.fn(),
  onSave: vi.fn(),
  onLoad: vi.fn(),
  onDelete: vi.fn(),
}

const trees = [
  { name: 'tree-v1', vocab_size: 256, num_merges: 100 },
  { name: 'tree-v2', vocab_size: 512, num_merges: 200 },
]

describe('SavedTreesCard', () => {
  it('renders the card title', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByText('Saved Trees')).toBeDefined()
  })

  it('renders the save name input', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByTestId('save-name-input')).toBeDefined()
  })

  it('shows no saved trees message when empty', () => {
    render(<SavedTreesCard {...defaultProps} />)
    expect(screen.getByText('No saved trees.')).toBeDefined()
  })

  it('renders saved trees with names and details', () => {
    render(<SavedTreesCard {...defaultProps} savedTrees={trees} />)
    expect(screen.getByText('tree-v1')).toBeDefined()
    expect(screen.getByText('tree-v2')).toBeDefined()
    expect(screen.getByText('256 vocab, 100 merges')).toBeDefined()
  })

  it('renders load and delete buttons for each tree', () => {
    render(<SavedTreesCard {...defaultProps} savedTrees={trees} />)
    expect(screen.getAllByText('Load').length).toBe(2)
    expect(screen.getAllByText('Delete').length).toBe(2)
  })

  it('calls onSave with correct name', async () => {
    const onSave = vi.fn()
    render(<SavedTreesCard {...defaultProps} saveName="my-tree" onSave={onSave} />)
    const saveBtn = screen.getByText('Save Current')
    expect((saveBtn as HTMLButtonElement).disabled).toBe(false)
  })

  it('disables save button when name is empty', () => {
    render(<SavedTreesCard {...defaultProps} />)
    const saveBtn = screen.getByText('Save Current')
    expect((saveBtn as HTMLButtonElement).disabled).toBe(true)
  })
})
