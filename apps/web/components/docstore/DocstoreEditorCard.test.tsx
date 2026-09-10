// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
  ),
}))

import { DocstoreEditorCard } from './DocstoreEditorCard'

afterEach(() => cleanup())

const defaultProps = {
  doc: { _id: 'doc-abc-123' },
  editMode: false,
  editContent: '{"key": "value"}',
  onEditToggle: vi.fn(),
  onEditContentChange: vi.fn(),
  onSave: vi.fn(),
  onDelete: vi.fn(),
}

describe('DocstoreEditorCard', () => {
  it('renders the title in empty state', () => {
    render(<DocstoreEditorCard {...defaultProps} doc={null} />)
    expect(screen.getByText('Document')).toBeTruthy()
  })

  it('shows empty state when doc is null', () => {
    render(<DocstoreEditorCard {...defaultProps} doc={null} />)
    expect(screen.getByText('No document selected.')).toBeTruthy()
  })

  it('shows document _id as title', () => {
    render(<DocstoreEditorCard {...defaultProps} />)
    expect(screen.getByText('doc-abc-123')).toBeTruthy()
  })

  it('shows Edit and Delete buttons when not editing', () => {
    render(<DocstoreEditorCard {...defaultProps} />)
    expect(screen.getByTestId('edit-btn')).toBeTruthy()
    expect(screen.getByTestId('delete-btn')).toBeTruthy()
    expect(screen.getByText('Edit')).toBeTruthy()
    expect(screen.getByText('Delete')).toBeTruthy()
  })

  it('shows Save and Cancel buttons when editing', () => {
    render(<DocstoreEditorCard {...defaultProps} editMode={true} />)
    expect(screen.getByTestId('save-btn')).toBeTruthy()
    expect(screen.getByTestId('cancel-btn')).toBeTruthy()
    expect(screen.getByText('Save')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('renders JSON content in pre when not editing', () => {
    render(<DocstoreEditorCard {...defaultProps} />)
    expect(screen.getByText('{"key": "value"}')).toBeTruthy()
  })

  it('renders textarea when editing', () => {
    render(<DocstoreEditorCard {...defaultProps} editMode={true} />)
    const textarea = screen.getByTestId('edit-textarea')
    expect(textarea).toBeTruthy()
    expect((textarea as HTMLTextAreaElement).value).toBe('{"key": "value"}')
  })

  it('calls onEditToggle when Edit clicked', () => {
    const onEditToggle = vi.fn()
    render(<DocstoreEditorCard {...defaultProps} onEditToggle={onEditToggle} />)
    fireEvent.click(screen.getByTestId('edit-btn'))
    expect(onEditToggle).toHaveBeenCalled()
  })

  it('calls onSave when Save clicked', () => {
    const onSave = vi.fn()
    render(<DocstoreEditorCard {...defaultProps} editMode={true} onSave={onSave} />)
    fireEvent.click(screen.getByTestId('save-btn'))
    expect(onSave).toHaveBeenCalled()
  })

  it('calls onDelete when Delete clicked', () => {
    const onDelete = vi.fn()
    render(<DocstoreEditorCard {...defaultProps} onDelete={onDelete} />)
    fireEvent.click(screen.getByTestId('delete-btn'))
    expect(onDelete).toHaveBeenCalled()
  })

  it('calls onEditContentChange when typing in textarea', () => {
    const onEditContentChange = vi.fn()
    render(<DocstoreEditorCard {...defaultProps} editMode={true} onEditContentChange={onEditContentChange} />)
    fireEvent.change(screen.getByTestId('edit-textarea'), { target: { value: 'new content' } })
    expect(onEditContentChange).toHaveBeenCalledWith('new content')
  })
})
