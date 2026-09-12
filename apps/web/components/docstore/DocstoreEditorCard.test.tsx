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
