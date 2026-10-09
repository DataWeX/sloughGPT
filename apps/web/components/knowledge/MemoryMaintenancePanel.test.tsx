// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
  FoldSection: ({ heading, children }: any) => <div><div>{heading}</div>{children}</div>,
  IconFilter: (props: any) => <span data-testid="icon-filter" {...props} />,
  IconFolder: (props: any) => <span data-testid="icon-folder" {...props} />,
  IconClock: (props: any) => <span data-testid="icon-clock" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {
    consolidate: vi.fn().mockResolvedValue({ removed: 0, kept: 0 }),
    archivePrune: vi.fn().mockResolvedValue({ pruned: 0 }),
    getConfig: vi.fn().mockResolvedValue({ archive_retention_days: 30 }),
    updateConfig: vi.fn().mockImplementation((cfg: any) => Promise.resolve(cfg)),
    list: vi.fn().mockResolvedValue({ items: [] }),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-01-01',
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  importFile: vi.fn(),
}))

vi.mock('@/lib/memory-card-utils', () => ({
  parseMemoryImport: vi.fn(() => []),
}))

vi.mock('@/components/SectionErrorBoundary', () => ({
  SectionErrorBoundary: ({ children }: any) => <div>{children}</div>,
}))

import { MemoryMaintenancePanel } from './MemoryMaintenancePanel'
import { memoryController } from '@/lib/memory-controller'

afterEach(() => cleanup())

const defaultProps = {
  itemCount: 5,
  archiveStats: { records: 10, bytes: 5120, path: 'mem:/archive', task_types: {}, oldest_ts: null, newest_ts: null },
  loading: false,
  fetchData: vi.fn().mockResolvedValue(undefined),
  openArchive: vi.fn(),
}

beforeEach(() => {
  vi.clearAllMocks()
  ;(memoryController.getConfig as any).mockResolvedValue({ archive_retention_days: 30 })
})

describe('MemoryMaintenancePanel', () => {
  it('renders without crashing', () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    expect(screen.getByText('Maintenance')).toBeTruthy()
  })

  it('renders consolidate section', () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    expect(screen.getByText('Consolidate duplicates')).toBeTruthy()
  })

  it('renders archive stats', async () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText(/10 record\(s\)/)).toBeTruthy()
    })
  })

  it('renders retention section', async () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText('Archive retention')).toBeTruthy()
    })
  })

  it('disables consolidate when itemCount is 0', () => {
    render(<MemoryMaintenancePanel {...defaultProps} itemCount={0} />)
    const btn = screen.getByRole('button', { name: /Consolidate/ })
    expect(btn).toHaveProperty('disabled', true)
  })

  it('disables prune when no archive records', async () => {
    render(<MemoryMaintenancePanel {...defaultProps} archiveStats={{ records: 0, bytes: 0, path: 'mem:/archive', task_types: {}, oldest_ts: null, newest_ts: null }} />)
    await waitFor(() => {
      const btn = screen.getByRole('button', { name: /Prune old/ })
      expect(btn).toHaveProperty('disabled', true)
    })
  })

  it('loads retention config on mount', async () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    await waitFor(() => {
      expect(memoryController.getConfig).toHaveBeenCalled()
    })
  })

  it('renders backup section', () => {
    render(<MemoryMaintenancePanel {...defaultProps} />)
    expect(screen.getByText('Backup memory')).toBeTruthy()
  })
})
