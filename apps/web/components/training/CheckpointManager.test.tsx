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
  Dialog: ({ open, children, onOpenChange }: any) => open ? <div data-testid="dialog">{children}</div> : null,
  DialogContent: ({ children, className }: any) => <div data-testid="dialog-content" className={className}>{children}</div>,
  DialogHeader: ({ children }: any) => <div data-testid="dialog-header">{children}</div>,
  DialogTitle: ({ children, className }: any) => <div data-testid="dialog-title" className={className}>{children}</div>,
  AlertDialog: ({ open, children, onOpenChange }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: any) => <div data-testid="alert-dialog-content">{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div data-testid="alert-dialog-title">{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
  AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
}))

function makeCheckpoints(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    name: `checkpoint-${i}`,
    soul: 'default',
    loss: 0.5 - i * 0.05,
    steps: 100 * (i + 1),
    epochs: i + 1,
    size_mb: 10 + i,
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

  it('renders nothing when no checkpoints', () => {
    const { container } = render(
      <CheckpointManager checkpoints={[]} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders checkpoint list', () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(3)} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.getByText('checkpoint-0')).toBeTruthy()
    expect(screen.getByText('checkpoint-1')).toBeTruthy()
    expect(screen.getByText('checkpoint-2')).toBeTruthy()
  })

  it('shows checkpoint count in title', () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(3)} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.getAllByText('Checkpoints (3)').length).toBeGreaterThanOrEqual(1)
  })

  it('calls loadCheckpoint when Load button clicked', async () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(2)} addToast={addToast} onRefresh={onRefresh} />
    )
    const loadButtons = screen.getAllByText('Load')
    fireEvent.click(loadButtons[0])
    await waitFor(() => {
      expect(mockController.loadCheckpoint).toHaveBeenCalledWith('checkpoint-0')
    })
    expect(addToast).toHaveBeenCalledWith('Loaded: checkpoint-0', 'success')
  })

  it('calls deleteCheckpoint when Delete button clicked', async () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(2)} addToast={addToast} onRefresh={onRefresh} />
    )
    const deleteButtons = screen.getAllByText('Delete')
    fireEvent.click(deleteButtons[0])
    await waitFor(() => {
      expect(mockController.deleteCheckpoint).toHaveBeenCalledWith('checkpoint-0')
    })
    expect(addToast).toHaveBeenCalledWith('Deleted: checkpoint-0', 'success')
    expect(onRefresh).toHaveBeenCalled()
  })

  it('shows info dialog when Info button clicked', async () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(1)} addToast={addToast} onRefresh={onRefresh} />
    )
    fireEvent.click(screen.getAllByText('Info')[0])
    await waitFor(() => {
      expect(mockController.getCheckpointInfo).toHaveBeenCalledWith('checkpoint-0')
    })
    expect(screen.getByTestId('dialog')).toBeTruthy()
    expect(screen.getAllByText('checkpoint-0').length).toBeGreaterThanOrEqual(1)
  })

  it('toggles individual checkpoint selection', () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(3)} addToast={addToast} onRefresh={onRefresh} />
    )
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    expect(checkboxes[0]).toBeChecked()
    fireEvent.click(checkboxes[0])
    expect(checkboxes[0]).not.toBeChecked()
  })

  it('select all and deselect all buttons exist', () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(3)} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.getAllByRole('button').some(b => b.textContent === 'Select all')).toBe(true)
  })

  it('shows delete button when checkbox selected', async () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(3)} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.queryByText(/Delete \d/)).toBeNull()
    const checkbox = screen.getAllByRole('checkbox')[0]
    fireEvent.click(checkbox)
    await waitFor(() => {
      expect(screen.getAllByText(/Delete 1/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('paginates checkpoints beyond PAGE_SIZE', async () => {
    const checkpoints = makeCheckpoints(15)
    render(
      <CheckpointManager checkpoints={checkpoints} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.getAllByText('checkpoint-0').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('checkpoint-10')).toBeNull()
    expect(screen.getAllByText(/1–10 of 15/).length).toBeGreaterThanOrEqual(1)
    const nextBtn = screen.getAllByRole('button').find(b => b.textContent === 'Next')!
    expect(nextBtn).not.toBeDisabled()
  })

  it('disables Prev on first page and Next on last page', () => {
    const checkpoints = makeCheckpoints(15)
    render(
      <CheckpointManager checkpoints={checkpoints} addToast={addToast} onRefresh={onRefresh} />
    )
    const prevBtns = screen.getAllByText('Prev')
    const nextBtns = screen.getAllByText('Next')
    expect(prevBtns[0]).toBeDisabled()
    expect(nextBtns[0]).not.toBeDisabled()
    fireEvent.click(nextBtns[0])
    expect(screen.getAllByText('Prev')[0]).not.toBeDisabled()
    expect(screen.getAllByText('Next')[0]).toBeDisabled()
  })

  it('shows checkpoint metadata', () => {
    render(
      <CheckpointManager checkpoints={makeCheckpoints(1)} addToast={addToast} onRefresh={onRefresh} />
    )
    expect(screen.getAllByText(/Loss 0.5000/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('100 steps').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1 epochs').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('10.0 MB').length).toBeGreaterThanOrEqual(1)
  })
})
