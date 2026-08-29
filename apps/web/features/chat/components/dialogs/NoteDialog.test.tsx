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
