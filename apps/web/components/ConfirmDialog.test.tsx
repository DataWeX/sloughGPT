import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

afterEach(cleanup)

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: passthrough,
    AlertDialogHeader: passthrough,
    AlertDialogTitle: ({ children }: any) => <div data-testid="dialog-title">{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div data-testid="dialog-desc">{children}</div>,
    AlertDialogFooter: passthrough,
    AlertDialogCancel: ({ children, onClick }: any) => <button aria-label="cancel" onClick={onClick}>{children}</button>,
    AlertDialogAction: ({ children, onClick }: any) => (
      <button aria-label="confirm" onClick={onClick}>{children}</button>
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
}
})

import { ConfirmDialog } from '@/components/ConfirmDialog'

describe('ConfirmDialog', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <ConfirmDialog open={false} onOpenChange={vi.fn()} title="Test" description="Desc" onConfirm={vi.fn()} />
    )
    expect(container.querySelector('[data-testid="alert-dialog"]')).toBeNull()
  })

  it('renders dialog when open', () => {
    render(
      <ConfirmDialog open={true} onOpenChange={vi.fn()} title="Confirm Action" description="Are you sure?" onConfirm={vi.fn()} />
    )
    expect(screen.getByText('Confirm Action')).toBeTruthy()
    expect(screen.getByText('Are you sure?')).toBeTruthy()
  })

  it('renders custom labels', () => {
    render(
      <ConfirmDialog open={true} onOpenChange={vi.fn()} title="Delete Item" description="Desc" onConfirm={vi.fn()}
        confirmLabel="Yes, delete" cancelLabel="No, keep" />
    )
    expect(screen.getByText('Yes, delete')).toBeTruthy()
    expect(screen.getByText('No, keep')).toBeTruthy()
  })

  it('calls onConfirm when confirm is clicked', () => {
    const onConfirm = vi.fn()
    render(
      <ConfirmDialog open={true} onOpenChange={vi.fn()} title="Delete Item" description="Desc" onConfirm={onConfirm} />
    )
    const btns = screen.getAllByRole('button', { name: 'confirm' })
    fireEvent.click(btns[btns.length - 1])
    expect(onConfirm).toHaveBeenCalled()
  })

  it('passes onOpenChange to AlertDialog', () => {
    const { container } = render(
      <ConfirmDialog open={true} onOpenChange={vi.fn()} title="Delete Item" description="Desc" onConfirm={vi.fn()} />
    )
    expect(container.querySelector('[data-testid="alert-dialog"]')).toBeTruthy()
  })
})
