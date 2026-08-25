// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { StopTrainingButton } from './StopTrainingButton'

vi.mock('@sloughgpt/strui', () => ({
  Button: ({ children, onClick, disabled, variant, size, className }: any) => (
    <button onClick={onClick} disabled={disabled} className={className} data-variant={variant} data-size={size}>{children}</button>
  ),
}))

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, title, description, confirmLabel }: any) => (
    open ? (
      <div data-testid="confirm-dialog">
        <div>{title}</div>
        <div>{description}</div>
        <button onClick={onConfirm}>{confirmLabel}</button>
      </div>
    ) : null
  ),
}))

describe('StopTrainingButton', () => {
  const mockOnStop = vi.fn()
  const mockToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockOnStop.mockResolvedValue(undefined)
  })

  it('renders stop button', () => {
    render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    expect(screen.getByText('Stop training')).toBeTruthy()
  })

  it('opens confirm dialog on click', async () => {
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    expect(screen.getByText('Stop training?')).toBeTruthy()
    unmount()
  })

  it('calls onStop when confirmed', async () => {
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(mockOnStop).toHaveBeenCalled()
    })
    expect(mockToast).toHaveBeenCalledWith('Training stopped', 'success')
    unmount()
  })

  it('shows error toast on failure', async () => {
    mockOnStop.mockRejectedValue(new Error('Network error'))
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith('Could not stop training', 'error')
    })
    unmount()
  })

  it('shows stopping state while processing', async () => {
    let resolveFn: () => void
    mockOnStop.mockImplementation(() => new Promise(r => { resolveFn = r }))
    const { unmount } = render(<StopTrainingButton onStop={mockOnStop} addToast={mockToast} />)
    fireEvent.click(screen.getAllByText('Stop training')[0])
    await waitFor(() => {
      expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
    })
    fireEvent.click(screen.getAllByText('Stop training')[1])
    await waitFor(() => {
      expect(screen.getAllByText('Stopping...').length).toBeGreaterThan(0)
    })
    resolveFn!()
    await waitFor(() => {
      expect(screen.queryByTestId('confirm-dialog')).toBeNull()
    })
    unmount()
  })
})
