import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => {
  function Dialog({ children, open }: any) { return open ? <div data-testid="dialog">{children}</div> : null }
  function DialogContent({ children }: any) { return <div data-testid="content">{children}</div> }
  function DialogHeader({ children }: any) { return <div>{children}</div> }
  function DialogTitle({ children }: any) { return <div>{children}</div> }
  function DialogPortal({ children }: any) { return <>{children}</> }
  function DialogOverlay() { return <div data-testid="overlay" /> }
  function Button({ children, onClick, ...props }: any) {
    return <button onClick={onClick} {...props}>{children}</button>
  }
  return {
    Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay,
    Button,
  
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

import { NoteDialog } from './NoteDialog'

const defaultProps = {
  open: true,
  onOpenChange: vi.fn(),
  note: '',
  onSave: vi.fn(),
  onDelete: vi.fn(),
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(cleanup)

describe('NoteDialog', () => {
  it('renders when open', async () => {
    render(<NoteDialog {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByTestId('dialog')).toBeDefined()
    })
    expect(screen.getByText('Add Note')).toBeDefined()
    expect(screen.getByLabelText('Message note')).toBeDefined()
  })

  it('does not render when closed', () => {
    render(<NoteDialog {...defaultProps} open={false} />)
    expect(screen.queryByTestId('dialog')).toBeNull()
  })

  it('shows Edit Note title when note exists', async () => {
    render(<NoteDialog {...defaultProps} note="Existing note" />)
    await waitFor(() => {
      expect(screen.getByText('Edit Note')).toBeDefined()
    })
  })

  it('displays existing note in textarea', async () => {
    render(<NoteDialog {...defaultProps} note="Existing note" />)
    await waitFor(() => {
      const textarea = screen.getByLabelText('Message note') as HTMLTextAreaElement
      expect(textarea.value).toBe('Existing note')
    })
  })

  it('calls onSave with draft text', async () => {
    const onSave = vi.fn()
    render(<NoteDialog {...defaultProps} onSave={onSave} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    const textarea = screen.getByLabelText('Message note')
    fireEvent.change(textarea, { target: { value: 'New note' } })
    fireEvent.click(screen.getByText('Save'))
    
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith('New note')
    })
  })

  it('calls onOpenChange(false) after save', async () => {
    const onOpenChange = vi.fn()
    render(<NoteDialog {...defaultProps} onOpenChange={onOpenChange} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    fireEvent.click(screen.getByText('Save'))
    
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it('shows toast on save', async () => {
    render(<NoteDialog {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    fireEvent.click(screen.getByText('Save'))
    
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Note saved', 'success')
    })
  })

  it('calls onDelete and shows toast when delete clicked', async () => {
    const onDelete = vi.fn()
    render(<NoteDialog {...defaultProps} onDelete={onDelete} note="Existing note" />)
    await waitFor(() => {
      expect(screen.getByText('Delete note')).toBeDefined()
    })
    
    fireEvent.click(screen.getByText('Delete note'))
    
    await waitFor(() => {
      expect(onDelete).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith('Note deleted', 'success')
    })
  })

  it('does not show delete button when no existing note', async () => {
    render(<NoteDialog {...defaultProps} note="" />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    expect(screen.queryByText('Delete note')).toBeNull()
  })

  it('does not show delete button when onDelete not provided', async () => {
    render(<NoteDialog {...defaultProps} note="Existing note" onDelete={undefined} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    expect(screen.queryByText('Delete note')).toBeNull()
  })

  it('calls onOpenChange(false) on cancel', async () => {
    const onOpenChange = vi.fn()
    render(<NoteDialog {...defaultProps} onOpenChange={onOpenChange} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    fireEvent.click(screen.getByText('Cancel'))
    
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('saves on Ctrl+Enter', async () => {
    const onSave = vi.fn()
    render(<NoteDialog {...defaultProps} onSave={onSave} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    const textarea = screen.getByLabelText('Message note')
    fireEvent.change(textarea, { target: { value: 'Quick note' } })
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true })
    
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith('Quick note')
    })
  })

  it('closes on Escape', async () => {
    const onOpenChange = vi.fn()
    render(<NoteDialog {...defaultProps} onOpenChange={onOpenChange} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    const textarea = screen.getByLabelText('Message note')
    fireEvent.keyDown(textarea, { key: 'Escape' })
    
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('trims whitespace from note', async () => {
    const onSave = vi.fn()
    render(<NoteDialog {...defaultProps} onSave={onSave} />)
    await waitFor(() => {
      expect(screen.getByLabelText('Message note')).toBeDefined()
    })
    
    const textarea = screen.getByLabelText('Message note')
    fireEvent.change(textarea, { target: { value: '  Trimmed note  ' } })
    fireEvent.click(screen.getByText('Save'))
    
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith('Trimmed note')
    })
  })

  it('saves empty string when clearing note', async () => {
    const onSave = vi.fn()
    render(<NoteDialog {...defaultProps} onSave={onSave} note="Old note" />)
    await waitFor(() => {
      const textarea = screen.getByLabelText('Message note') as HTMLTextAreaElement
      expect(textarea.value).toBe('Old note')
    })
    
    const textarea = screen.getByLabelText('Message note')
    fireEvent.change(textarea, { target: { value: '' } })
    fireEvent.click(screen.getByText('Save'))
    
    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith('')
    })
  })
})
