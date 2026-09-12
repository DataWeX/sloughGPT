import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { VectorStoreInitCard } from './VectorStoreInitCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, variant, ...props }: any) => (
    <button data-testid={`btn-${variant || 'default'}`} onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),

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
  provider: 'in_memory',
  initializing: false,
  onInit: vi.fn(),
}

describe('VectorStoreInitCard', () => {
  it('renders the card title', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText('Initialize')).toBeDefined()
  })

  it('renders the description text', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText(/Choose a vector store backend/)).toBeDefined()
  })

  it('renders In Memory and ChromaDB buttons', () => {
    render(<VectorStoreInitCard {...defaultProps} />)
    expect(screen.getByText('In Memory')).toBeDefined()
    expect(screen.getByText('ChromaDB')).toBeDefined()
  })

  it('calls onInit with in_memory when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorStoreInitCard {...defaultProps} onInit={onInit} />)
    await userEvent.click(screen.getByText('In Memory'))
    expect(onInit).toHaveBeenCalledWith('in_memory')
  })

  it('calls onInit with chromadb when clicked', async () => {
    const onInit = vi.fn()
    render(<VectorStoreInitCard {...defaultProps} onInit={onInit} />)
    await userEvent.click(screen.getByText('ChromaDB'))
    expect(onInit).toHaveBeenCalledWith('chromadb')
  })

  it('disables buttons when initializing', () => {
    render(<VectorStoreInitCard {...defaultProps} initializing={true} />)
    expect((screen.getByText('In Memory') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('ChromaDB') as HTMLButtonElement).disabled).toBe(true)
  })
})
