import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()
const mockApiPut = vi.fn()
const mockApiDelete = vi.fn()
const mockApiPatch = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPut: (...args: unknown[]) => mockApiPut(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
  apiPatch: (...args: unknown[]) => mockApiPatch(...args),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, className, id }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} id={id} />
    ),
    Label: ({ children, htmlFor, variant }: any) => <label htmlFor={htmlFor} data-variant={variant}>{children}</label>,
    AlertDialog: ({ children, open }: any) => open ? <div role="dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
    AlertDialogContent: passthrough,
    AlertDialogDescription: passthrough,
    AlertDialogFooter: passthrough,
    AlertDialogHeader: passthrough,
    AlertDialogTitle: passthrough,
  
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
}
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import DocstorePage from './page'

describe('DocstorePage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    expect(screen.getByText('Document store')).toBeInTheDocument()
    expect(screen.getByText('Browse and manage stored documents')).toBeInTheDocument()
  })

  it('fetches documents on mount', async () => {
    mockApiGet.mockResolvedValue({ documents: [{ _id: 'doc1' }, { _id: 'doc2' }], total: 2 })
    render(<DocstorePage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/docstore/sessions', expect.anything())
    }, { timeout: 5000 })
  })

  it('displays documents', async () => {
    mockApiGet.mockResolvedValue({ documents: [{ _id: 'doc1' }, { _id: 'doc2' }], total: 2 })
    render(<DocstorePage />)
    await waitFor(() => {
      expect(screen.getByText('doc1')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('doc2')).toBeInTheDocument()
  })

  it('switches collection when clicking a collection button', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalled()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('knowledge'))
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledWith('/docstore/knowledge', expect.anything())
    }, { timeout: 5000 })
  })

  it('shows empty state when no documents', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    await waitFor(() => {
      expect(screen.getByText('No documents.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('shows error toast on fetch failure', async () => {
    mockApiGet.mockRejectedValue(new Error('network'))
    render(<DocstorePage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load sessions', 'error')
    }, { timeout: 5000 })
  })

  it('shows create form when New doc clicked', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    fireEvent.click(screen.getByText('New doc'))
    expect(screen.getByText('Create document in sessions')).toBeInTheDocument()
  })

  it('creates a document', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    mockApiPut.mockResolvedValue({})
    render(<DocstorePage />)
    fireEvent.click(screen.getByText('New doc'))
    fireEvent.change(screen.getByPlaceholderText('my-doc-id'), { target: { value: 'new-doc' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(mockApiPut).toHaveBeenCalledWith('/docstore/sessions/new-doc', {})
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Document created', 'success')
  })

  it('validates document ID is required', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    fireEvent.click(screen.getByText('New doc'))
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Document ID is required', 'error')
    }, { timeout: 5000 })
  })

  it('validates JSON content', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    fireEvent.click(screen.getByText('New doc'))
    fireEvent.change(screen.getByPlaceholderText('my-doc-id'), { target: { value: 'doc1' } })
    fireEvent.change(screen.getByDisplayValue('{}'), { target: { value: 'not-json' } })
    fireEvent.click(screen.getByText('Create'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Invalid JSON', 'error')
    }, { timeout: 5000 })
  })

  it('deletes a document', async () => {
    mockApiGet.mockResolvedValue({ documents: [{ _id: 'doc1' }], total: 1 })
    mockApiDelete.mockResolvedValue({})
    render(<DocstorePage />)
    await waitFor(() => {
      expect(screen.getByText('doc1')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('doc1'))
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => {
      expect(mockApiDelete).toHaveBeenCalledWith('/docstore/sessions/doc1')
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Document deleted', 'success')
  })

  it('refreshes on Refresh click', async () => {
    mockApiGet.mockResolvedValue({ documents: [], total: 0 })
    render(<DocstorePage />)
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockApiGet).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
