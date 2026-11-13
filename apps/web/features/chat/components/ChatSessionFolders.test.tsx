import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconPlus: (props: any) => <span data-testid="icon-plus" {...props} />,
  IconCheck: (props: any) => <span data-testid="icon-check" {...props} />,

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

import { ChatSessionFolders } from './ChatSessionFolders'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockSessions = [
  { id: '1', title: 'Chat 1', folderId: null },
  { id: '2', title: 'Chat 2', folderId: null },
  { id: '3', title: 'Chat 3', folderId: 'f1' },
]

describe('ChatSessionFolders', () => {
  it('renders empty state', () => {
    render(<ChatSessionFolders sessions={[]} onMoveSession={vi.fn()} />)
    expect(screen.getByText('No folders')).toBeInTheDocument()
  })

  it('renders unfiled sessions count', () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    expect(screen.getByText(/Unfiled/)).toBeInTheDocument()
  })

  it('creates a folder', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Folder name...'), { target: { value: 'My Folder' } })
    await act(async () => {
      fireEvent.click(screen.getByTestId('icon-check').closest('button')!)
    })
    expect(screen.getByText('My Folder')).toBeInTheDocument()
  })

  it('creates folder on Enter key', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    const input = screen.getByPlaceholderText('Folder name...')
    fireEvent.change(input, { target: { value: 'Test' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getByText('Test')).toBeInTheDocument()
  })

  it('persists folders to localStorage', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Folder name...'), { target: { value: 'Saved' } })
    await act(async () => {
      fireEvent.click(screen.getByTestId('icon-check').closest('button')!)
    })
    const stored = JSON.parse(localStorage.getItem('chat-folders') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Saved')
  })

  it('toggles folder expansion', async () => {
    const sessionsWithFolder = [
      { id: '1', title: 'Chat in folder', folderId: 'f1' },
    ]
    localStorage.setItem('chat-folders', JSON.stringify([{ id: 'f1', name: 'Test Folder', createdAt: Date.now() }]))
    render(<ChatSessionFolders sessions={sessionsWithFolder} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByText('Test Folder'))
    expect(screen.getByText('Chat in folder')).toBeInTheDocument()
  })

  it('deletes a folder and moves sessions to unfiled', async () => {
    const onMoveSession = vi.fn()
    const sessionsWithFolder = [
      { id: '1', title: 'Chat in folder', folderId: 'f1' },
    ]
    localStorage.setItem('chat-folders', JSON.stringify([{ id: 'f1', name: 'To Delete', createdAt: Date.now() }]))
    render(<ChatSessionFolders sessions={sessionsWithFolder} onMoveSession={onMoveSession} />)
    fireEvent.click(screen.getByTitle('Delete folder'))
    expect(onMoveSession).toHaveBeenCalledWith('1', null)
  })

  it('disables save when name empty', () => {
    render(<ChatSessionFolders sessions={[]} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    const saveBtn = screen.getByTestId('icon-check').closest('button')
    expect(saveBtn).toBeDefined()
    expect(saveBtn).toBeDisabled()
  })
})