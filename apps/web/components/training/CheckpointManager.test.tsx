// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { CheckpointManager } from './CheckpointManager'

const { mockController } = vi.hoisted(() => ({
  mockController: {
    loadCheckpoint: vi.fn().mockResolvedValue({ success: true }),
    deleteCheckpoint: vi.fn().mockResolvedValue({ success: true }),
    deleteCheckpointsBatch: vi.fn().mockResolvedValue({ deleted: 2 }),
    downloadCheckpoint: vi.fn().mockResolvedValue(new Blob(['test'])),
    getCheckpointInfo: vi.fn().mockResolvedValue({ name: 'cp-1', created: '2024-01-01' }),
  },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: mockController,
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, className }: any) => <div data-testid="card" className={className}>{children}</div>,
  CardHeader: ({ children, className }: any) => <div data-testid="card-header" className={className}>{children}</div>,
  CardTitle: ({ children, className }: any) => <div data-testid="card-title" className={className}>{children}</div>,
  CardContent: ({ children, className }: any) => <div data-testid="card-content" className={className}>{children}</div>,
  Button: ({ children, onClick, disabled, variant, className }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} data-variant={variant}>{children}</button>
  ),
  cn: (...classes: (string | false | undefined)[]) => classes.filter(Boolean).join(' '),
  Dialog: ({ open, children }: any) => open ? <div data-testid="dialog">{children}</div> : null,
  DialogContent: ({ children, className }: any) => <div data-testid="dialog-content" className={className}>{children}</div>,
  DialogHeader: ({ children }: any) => <div data-testid="dialog-header">{children}</div>,
  DialogTitle: ({ children, className }: any) => <div data-testid="dialog-title" className={className}>{children}</div>,
  AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: any) => <div data-testid="alert-dialog-content">{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div data-testid="alert-dialog-title">{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
  AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
}))

function makeCps(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    name: `cp-${i}`, soul: 'default', loss: 0.5 - i * 0.05,
    steps: 100 * (i + 1), epochs: i + 1, size_mb: 10 + i,
  }))
}

describe('CheckpointManager', () => {
  const addToast = vi.fn()
  const onRefresh = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    addToast.mockReset()
    onRefresh.mockReset()
    mockController.loadCheckpoint.mockResolvedValue({ success: true })
    mockController.deleteCheckpoint.mockResolvedValue({ success: true })
    mockController.deleteCheckpointsBatch.mockResolvedValue({ deleted: 2 })
    mockController.getCheckpointInfo.mockResolvedValue({ name: 'cp-1', created: '2024-01-01' })
  })

  it('renders nothing when empty', () => {
    const { container } = render(<CheckpointManager checkpoints={[]} addToast={addToast} onRefresh={onRefresh} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders checkpoint list', () => {
    render(<CheckpointManager checkpoints={makeCps(3)} addToast={addToast} onRefresh={onRefresh} />)
    expect(screen.getByText('cp-0')).toBeTruthy()
    expect(screen.getByText('cp-1')).toBeTruthy()
    expect(screen.getByText('cp-2')).toBeTruthy()
  })

  it('shows count in title', () => {
    render(<CheckpointManager checkpoints={makeCps(3)} addToast={addToast} onRefresh={onRefresh} />)
    expect(screen.getAllByText('Checkpoints (3)').length).toBeGreaterThanOrEqual(1)
  })

  it('loads checkpoint', async () => {
    render(<CheckpointManager checkpoints={makeCps(2)} addToast={addToast} onRefresh={onRefresh} />)
    fireEvent.click(screen.getAllByText('Load')[0])
    await waitFor(() => expect(mockController.loadCheckpoint).toHaveBeenCalledWith('cp-0'))
    expect(addToast).toHaveBeenCalledWith('Loaded: cp-0', 'success')
  })

  it('deletes checkpoint', async () => {
    render(<CheckpointManager checkpoints={makeCps(2)} addToast={addToast} onRefresh={onRefresh} />)
    fireEvent.click(screen.getAllByText('Delete')[0])
    await waitFor(() => expect(mockController.deleteCheckpoint).toHaveBeenCalledWith('cp-0'))
    expect(addToast).toHaveBeenCalledWith('Deleted: cp-0', 'success')
    expect(onRefresh).toHaveBeenCalled()
  })

  it('shows info dialog', async () => {
    render(<CheckpointManager checkpoints={makeCps(1)} addToast={addToast} onRefresh={onRefresh} />)
    fireEvent.click(screen.getAllByText('Info')[0])
    await waitFor(() => expect(mockController.getCheckpointInfo).toHaveBeenCalledWith('cp-0'))
    expect(screen.getByTestId('dialog')).toBeTruthy()
  })

  it('toggles checkbox selection', () => {
    render(<CheckpointManager checkpoints={makeCps(3)} addToast={addToast} onRefresh={onRefresh} />)
    const cb = screen.getAllByRole('checkbox')[0]
    fireEvent.click(cb)
    expect(cb).toBeChecked()
    fireEvent.click(cb)
    expect(cb).not.toBeChecked()
  })

  it('shows select all and delete count', () => {
    render(<CheckpointManager checkpoints={makeCps(3)} addToast={addToast} onRefresh={onRefresh} />)
    expect(screen.getAllByText('Select all').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText(/Delete \d/)).toBeNull()
    fireEvent.click(screen.getAllByRole('checkbox')[0])
    expect(screen.getByText('Delete 1')).toBeTruthy()
  })

  it('paginates when more than PAGE_SIZE', () => {
    render(<CheckpointManager checkpoints={makeCps(15)} addToast={addToast} onRefresh={onRefresh} />)
    expect(screen.getAllByText('cp-0').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('cp-10')).toBeNull()
    expect(screen.getByText('1–10 of 15')).toBeTruthy()
    expect(screen.getAllByText('Next').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Prev').length).toBeGreaterThanOrEqual(1)
  })

  it('shows checkpoint metadata', () => {
    render(<CheckpointManager checkpoints={makeCps(1)} addToast={addToast} onRefresh={onRefresh} />)
    expect(screen.getAllByText(/Loss 0.5000/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('100 steps').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1 epochs').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('10.0 MB').length).toBeGreaterThanOrEqual(1)
  })
})
