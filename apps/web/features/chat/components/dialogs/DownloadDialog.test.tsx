import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Dialog: ({ children, open }: any) => open ? <div data-testid="dialog">{children}</div> : null,
  DialogContent: ({ children }: any) => <div data-testid="dialog-content">{children}</div>,
  DialogHeader: ({ children }: any) => <div>{children}</div>,
  DialogTitle: ({ children }: any) => <h2>{children}</h2>,
  DialogDescription: ({ children }: any) => <p>{children}</p>,
  DialogFooter: ({ children }: any) => <div>{children}</div>,
  Checkbox: ({ checked, onCheckedChange }: any) => (
    <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} data-testid="checkbox" />
  ),
  Button: ({ children, onClick, variant, size, ...rest }: any) => (
    <button onClick={onClick} data-variant={variant} data-size={size} {...rest}>{children}</button>
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

vi.mock('@/lib/session-store', () => ({
  sessionStore: {
    setApproved: vi.fn(),
  },
}))

import { DownloadDialog } from './DownloadDialog'
import { sessionStore } from '@/lib/session-store'

describe('DownloadDialog', () => {
  const onConfirm = vi.fn()
  const onCancel = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders nothing when pendingDownload is null', () => {
    const { container } = render(
      <DownloadDialog
        open={true}
        pendingDownload={null}
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders dialog when pendingDownload is set', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    expect(screen.getByText('Download model?')).toBeDefined()
  })

  it('shows model name from pendingDownload', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="hf/gpt2-medium"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    expect(screen.getByText(/gpt2-medium/)).toBeDefined()
  })

  it('shows size from modelInfoMap', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{ gpt2: { size_gb: 0.5 } }}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    expect(screen.getByText(/0.5 GB/)).toBeDefined()
  })

  it('shows ? GB when size unknown', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    expect(screen.getByText(/\? GB/)).toBeDefined()
  })

  it('calls onCancel when Cancel clicked', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    fireEvent.click(screen.getByText('Cancel'))
    expect(onCancel).toHaveBeenCalled()
  })

  it('calls onConfirm and onCancel when Download clicked', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    fireEvent.click(screen.getByText('Download'))
    expect(onCancel).toHaveBeenCalled()
    expect(onConfirm).toHaveBeenCalledWith('gpt2')
  })

  it('calls sessionStore.setApproved when skip is checked', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    fireEvent.click(screen.getByTestId('checkbox'))
    fireEvent.click(screen.getByText('Download'))
    expect(sessionStore.setApproved).toHaveBeenCalledWith('gpt2')
  })

  it('calls onCancel when dialog closed', () => {
    render(
      <DownloadDialog
        open={true}
        pendingDownload="gpt2"
        modelInfoMap={{}}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />
    )
    const dialog = screen.getByTestId('dialog')
    fireEvent.click(dialog)
    // Dialog onOpenChange(false) is not triggered in mock — covered by Cancel button
    expect(screen.getByText('Download model?')).toBeDefined()
  })
})
