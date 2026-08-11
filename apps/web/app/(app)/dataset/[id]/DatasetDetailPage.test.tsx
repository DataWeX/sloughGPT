import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

// ── cva mock ──
const mockCva = vi.hoisted(() => { const fn = () => ''; return fn })
vi.mock('class-variance-authority', () => ({ cva: () => mockCva }))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough, CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, variant, size, className, disabled, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} aria-label={ariaLabel} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, className, placeholder, 'aria-label': ariaLabel }: any) => (
      <input value={value} onChange={onChange} className={className} placeholder={placeholder} aria-label={ariaLabel} />
    ),
    Badge: ({ children, variant, className }: any) => <span data-variant={variant} className={className}>{children}</span>,
    StatCard: ({ label, value, className }: any) => <div className={className}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    Skeleton: ({ className }: any) => <div className={className} />,
    IconTrash: iconMock('trash'), IconDownload: iconMock('download'), IconEdit: iconMock('edit'),
    IconCheck: iconMock('check'), IconX: iconMock('x'), IconRefresh: iconMock('refresh'), IconClock: iconMock('clock'),
    IconChevronDown: iconMock('chevron-down'),
    AlertDialog: ({ open, onOpenChange, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
    Breadcrumbs: ({ items }: any) => <nav aria-label="Breadcrumb">{items?.map((item: any, i: number) => <span key={i}>{item.label}</span>)}</nav>,
  }
})

// ── controller & router mocks ──
const { mockGet, mockUpdate, mockDelete, mockExport, mockGetStats, mockPush, mockAddToast, mockListVersions, mockCreateVersion, mockRestoreVersion, mockConvertToMessages } = vi.hoisted(() => ({
  mockGet: vi.fn(), mockUpdate: vi.fn(), mockDelete: vi.fn(),
  mockExport: vi.fn(), mockGetStats: vi.fn(), mockPush: vi.fn(), mockAddToast: vi.fn(),
  mockListVersions: vi.fn(), mockCreateVersion: vi.fn(), mockRestoreVersion: vi.fn(),
  mockConvertToMessages: vi.fn(),
}))
const stableRouter = { push: mockPush }

vi.mock('next/navigation', () => ({ useRouter: () => stableRouter, useParams: () => ({ id: 'shakespeare' }) }))
vi.mock('@/lib/dataset-controller', () => ({ datasetController: { get: mockGet, update: mockUpdate, delete: mockDelete, export: mockExport, getStats: mockGetStats, listVersions: mockListVersions, createVersion: mockCreateVersion, restoreVersion: mockRestoreVersion, convertToMessages: mockConvertToMessages } }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/components/DatasetPreview', () => ({ DatasetPreview: () => null }))
vi.stubGlobal('URL', { createObjectURL: vi.fn(), revokeObjectURL: vi.fn() })

import DatasetDetailPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockListVersions.mockResolvedValue({ versions: [], count: 0 })
  mockGetStats.mockResolvedValue({
    format: 'jsonl',
    samples: 500,
    chars: 120000,
    avg_length: 240,
    has_messages: false,
    sample_preview: [],
    lines: 500,
    suggested_method: 'finetune',
    file_type: 'jsonl',
  })
})

const mockDataset = {
  id: 'shakespeare', name: 'Shakespeare Works', type: 'text',
  source: 'local', size: 1048576, samples: 500,
  tags: ['literature'], created_at: '2026-06-01T12:00:00Z',
}

function waitForName() {
  return waitFor(() => { expect(screen.getAllByText('Shakespeare Works').length).toBeGreaterThan(0) })
}

describe('DatasetDetailPage', () => {

  it('shows loading initially and calls get', () => {
    mockGet.mockReturnValue(new Promise(() => {}))
    render(<DatasetDetailPage />)
    expect(screen.getAllByText('...').length).toBeGreaterThan(0)
    expect(mockGet).toHaveBeenCalledWith('shakespeare')
  })

  it('displays dataset name after loading', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
  })

  it('shows not-found card on fetch failure', async () => {
    mockGet.mockRejectedValueOnce(new Error('not found'))
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('Dataset not found')).toBeTruthy() })
  })

  it('shows stat cards with correct values', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('text')).toBeTruthy() })
    expect(screen.getAllByText('500').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('1.0 MB')).toBeTruthy()
    expect(screen.getByText('local')).toBeTruthy()
    expect(screen.getByText('literature')).toBeTruthy()
    expect(screen.getByText('Preview')).toBeTruthy()
  })

  it('shows rename input on rename button click', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    expect(screen.getByDisplayValue('Shakespeare Works')).toBeTruthy()
  })

  it('commits rename on confirm', async () => {
    mockUpdate.mockResolvedValue({})
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    const input = screen.getByDisplayValue('Shakespeare Works')
    await act(async () => { fireEvent.change(input, { target: { value: 'Renamed' } }) })
    await act(async () => { screen.getByLabelText('Confirm rename').click() })
    await waitFor(() => { expect(mockUpdate).toHaveBeenCalledWith('shakespeare', { name: 'Renamed' }) })
  })

  it('cancels rename on cancel', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Rename dataset').click() })
    await act(async () => { screen.getByLabelText('Cancel rename').click() })
    expect(mockUpdate).not.toHaveBeenCalled()
  })

  it('deletes dataset on delete with confirmation', async () => {
    mockDelete.mockResolvedValue({})
    mockGet.mockResolvedValue(mockDataset)
    const { container } = render(<DatasetDetailPage />)
    await waitForName()
    const deleteBtn = container.querySelector('button.text-destructive') as HTMLElement
    await act(async () => { deleteBtn.click() })
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })
    const dialog = screen.getByTestId('alert-dialog')
    const confirmBtn = dialog.querySelector('button:last-child') as HTMLElement
    await act(async () => { confirmBtn.click() })
    await waitFor(() => { expect(mockDelete).toHaveBeenCalledWith('shakespeare') })
    expect(mockPush).toHaveBeenCalledWith('/datasets')
  })

  it('exports dataset on export button click', async () => {
    mockExport.mockResolvedValue(new Blob())
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByText('Export').click() })
    await act(async () => { screen.getByText('Export as JSONL').click() })
    await waitFor(() => { expect(mockExport).toHaveBeenCalledWith('shakespeare') })
  })

  it('fetches and displays dataset stats', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitFor(() => { expect(screen.getByText('Stats')).toBeTruthy() })
    expect(mockGetStats).toHaveBeenCalledWith('shakespeare')
    expect(screen.getByText('jsonl')).toBeTruthy()
    expect(screen.getAllByText('500').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('240 chars')).toBeTruthy()
    expect(screen.getByText('120,000')).toBeTruthy()
    expect(screen.getByText('finetune')).toBeTruthy()
  })

  it('does not show Stats card when stats fail', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockGetStats.mockRejectedValueOnce(new Error('stats unavailable'))
    render(<DatasetDetailPage />)
    await waitForName()
    await waitFor(() => { expect(mockGetStats).toHaveBeenCalledWith('shakespeare') })
    expect(screen.queryByText('Stats')).toBeFalsy()
  })

  it('shows empty versions state', async () => {
    mockGet.mockResolvedValue(mockDataset)
    render(<DatasetDetailPage />)
    await waitForName()
    await waitFor(() => { expect(screen.getByText('Versions')).toBeTruthy() })
    expect(mockListVersions).toHaveBeenCalledWith('shakespeare')
    expect(screen.getByText(/No snapshots yet/)).toBeTruthy()
  })

  it('lists existing versions', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockListVersions.mockResolvedValue({ versions: ['20260801120000'], count: 1 })
    render(<DatasetDetailPage />)
    await waitForName()
    await waitFor(() => { expect(screen.getByText('20260801120000')).toBeTruthy() })
  })

  it('creates a snapshot on button click', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockCreateVersion.mockResolvedValue({ timestamp: '20260801120000', message: 'Version created' })
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByText('Create snapshot').click() })
    await waitFor(() => { expect(mockCreateVersion).toHaveBeenCalledWith('shakespeare') })
  })

  it('restores a version after confirmation', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockRestoreVersion.mockResolvedValue({ success: true, message: 'Version restored' })
    mockListVersions.mockResolvedValue({ versions: ['20260801120000'], count: 1 })
    render(<DatasetDetailPage />)
    await waitForName()
    await waitFor(() => { expect(screen.getByText('20260801120000')).toBeTruthy() })
    await act(async () => { screen.getByText('Restore').click() })
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })
    const dialog = screen.getByTestId('alert-dialog')
    const confirmBtn = dialog.querySelector('button:last-child') as HTMLElement
    await act(async () => { confirmBtn.click() })
    await waitFor(() => { expect(mockRestoreVersion).toHaveBeenCalledWith('shakespeare', '20260801120000') })
  })

  it('converts dataset to chat format with default system prompt', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockConvertToMessages.mockResolvedValue({ status: 'converted', new_dataset_id: 'shakespeare-messages', total_conversations: 3 })
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByRole('button', { name: /Convert to chat format/ }).click() })
    await waitFor(() => { expect(mockConvertToMessages).toHaveBeenCalledWith('shakespeare', 'You are a helpful assistant.') })
    expect(screen.getByText(/Created Shakespeare Works-messages with 3 conversations/)).toBeTruthy()
  })

  it('uses a custom system prompt when provided', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockConvertToMessages.mockResolvedValue({ status: 'converted', new_dataset_id: 'shakespeare-messages', total_conversations: 1 })
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { fireEvent.change(screen.getByLabelText('System prompt'), { target: { value: 'You are a poet.' } }) })
    await act(async () => { screen.getByRole('button', { name: /Convert to chat format/ }).click() })
    await waitFor(() => { expect(mockConvertToMessages).toHaveBeenCalledWith('shakespeare', 'You are a poet.') })
    expect(screen.getByText(/1 conversation/)).toBeTruthy()
  })

  it('shows an error toast when conversion fails', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockConvertToMessages.mockRejectedValueOnce(new Error('conversion failed'))
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByRole('button', { name: /Convert to chat format/ }).click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Conversion failed', 'error') })
    expect(screen.queryByText('Open converted dataset')).toBeFalsy()
  })

  it('opens the converted dataset from the result banner', async () => {
    mockGet.mockResolvedValue(mockDataset)
    mockConvertToMessages.mockResolvedValue({ status: 'converted', new_dataset_id: 'shakespeare-messages', total_conversations: 3 })
    render(<DatasetDetailPage />)
    await waitForName()
    await act(async () => { screen.getByRole('button', { name: /Convert to chat format/ }).click() })
    await waitFor(() => { expect(screen.getByText('Open converted dataset')).toBeTruthy() })
    await act(async () => { screen.getByText('Open converted dataset').click() })
    expect(mockPush).toHaveBeenCalledWith('/dataset/shakespeare-messages')
  })
})
