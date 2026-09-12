// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  EmptyCard: ({ message }: any) => <div data-testid="empty-card">{message}</div>,
  Dialog: ({ open, children, ...props }: any) => open ? <div data-testid="dialog">{children}</div> : null,
  DialogContent: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  DialogDescription: ({ children }: any) => <div>{children}</div>,
  DialogFooter: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  DialogHeader: ({ children }: any) => <div>{children}</div>,
  DialogTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  IconFolder: (props: any) => <span data-testid="icon-folder" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,
  IconChevronDown: (props: any) => <span data-testid="icon-chevron" {...props} />,
  Spinner: (props: any) => <span data-testid="spinner" {...props} />,
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: { archive: vi.fn().mockResolvedValue({ records: [] }) },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-01-01',
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/components/SectionErrorBoundary', () => ({
  SectionErrorBoundary: ({ children }: any) => <div>{children}</div>,
}))

vi.mock('@/lib/memory-card-utils', () => ({
  archiveTypeLabel: (type: string) => type,
  archiveBadgeClass: () => 'bg-primary/80',
  archiveSummary: (record: any) => ({ text: record.task_type || '—', detail: null }),
}))

import { ArchiveDialog } from './ArchiveDialog'
import { memoryController } from '@/lib/memory-controller'

afterEach(() => cleanup())

const defaultProps = {
  open: true,
  onOpenChange: vi.fn(),
  archiveStats: { records: 5, bytes: 10240 },
}

beforeEach(() => {
  vi.clearAllMocks()
  ;(memoryController.archive as any).mockResolvedValue({ records: [] })
})

describe('ArchiveDialog', () => {
  it('renders without crashing', () => {
    render(<ArchiveDialog {...defaultProps} />)
    expect(screen.getAllByText('Provenance archive').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty card when no records', async () => {
    ;(memoryController.archive as any).mockResolvedValue({ records: [] })
    render(<ArchiveDialog {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByTestId('empty-card')).toBeTruthy()
    })
  })

  it('shows archive stats in description', () => {
    render(<ArchiveDialog {...defaultProps} />)
    expect(screen.getByText('5 record(s) — 10.0 KB')).toBeTruthy()
  })

  it('shows loading skeletons while loading', () => {
    ;(memoryController.archive as any).mockReturnValue(new Promise(() => {}))
    render(<ArchiveDialog {...defaultProps} />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('renders archive records', async () => {
    ;(memoryController.archive as any).mockResolvedValue({
      records: [
        { task_id: '1', task_type: 'store', ts: 1700000000 },
        { task_id: '2', task_type: 'consolidate', ts: 1700001000 },
      ],
    })
    render(<ArchiveDialog {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getAllByText('store').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('consolidate').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('calls onOpenChange when close button clicked', () => {
    const onOpenChange = vi.fn()
    render(<ArchiveDialog {...defaultProps} onOpenChange={onOpenChange} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
