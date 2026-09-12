/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { ApiKeyListCard } from './ApiKeyListCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
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
}))

vi.mock('@/components/icons/NavIcons', () => ({
  IconRefresh: (props: any) => <svg data-testid="icon-refresh" {...props} />,
  IconTrash: (props: any) => <svg data-testid="icon-trash" {...props} />,
}))

describe('ApiKeyListCard', () => {
  const mockKeys = [
    { id: '1', name: 'Test Key', key_hash: 'sk_abc123', scopes: ['*'], created_at: 1700000000, revoked: false },
    { id: '2', name: 'Old Key', key_hash: 'sk_xyz789', scopes: ['read'], created_at: 1690000000, revoked: true },
  ]

  it('renders the title', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('Active Keys')).toBeDefined()
  })

  it('renders custom title', () => {
    render(<ApiKeyListCard keys={[]} title="Revoked Keys" />)
    expect(screen.getByText('Revoked Keys')).toBeDefined()
  })

  it('renders empty message when no keys', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('No API keys.')).toBeDefined()
  })

  it('renders key name when keys provided', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('Test Key')).toBeDefined()
  })

  it('renders key hash', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('sk_abc123')).toBeDefined()
  })

  it('calls onRotate when rotate button clicked', () => {
    const onRotate = vi.fn()
    render(<ApiKeyListCard keys={[mockKeys[0]]} onRotate={onRotate} />)
    fireEvent.click(screen.getByTitle('Rotate key'))
    expect(onRotate).toHaveBeenCalledWith('1')
  })

  it('calls onRevoke when revoke button clicked', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={[mockKeys[0]]} onRevoke={onRevoke} />)
    fireEvent.click(screen.getByTitle('Revoke key'))
    expect(onRevoke).toHaveBeenCalledWith('1')
  })

  it('shows Revoked badge for revoked keys', () => {
    render(<ApiKeyListCard keys={[mockKeys[1]]} />)
    expect(screen.getByText('Revoked')).toBeDefined()
  })
})
