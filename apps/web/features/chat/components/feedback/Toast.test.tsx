/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ToastContainer } from './Toast'

afterEach(cleanup)

describe('ToastContainer', () => {
  it('renders nothing when toasts array is empty', () => {
    const { container } = render(<ToastContainer toasts={[]} onDismiss={() => {}} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders a toast with message and type', () => {
    render(<ToastContainer toasts={[{ id: '1', message: 'Done!', type: 'success' }]} onDismiss={() => {}} />)
    expect(screen.getByText('Done!')).toBeInTheDocument()
  })

  it('renders all toast types', () => {
    render(
      <ToastContainer
        toasts={[
          { id: '1', message: 'Success', type: 'success' },
          { id: '2', message: 'Error', type: 'error' },
          { id: '3', message: 'Info', type: 'info' },
        ]}
        onDismiss={() => {}}
      />
    )
    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('Info')).toBeInTheDocument()
  })

  it('renders verbose content when toggled', async () => {
    const user = userEvent.setup()
    render(
      <ToastContainer
        toasts={[{ id: '1', message: 'Saved', type: 'info', verbose: 'Saved 5 files to disk' }]}
        onDismiss={() => {}}
      />
    )
    await user.click(screen.getByText('Details'))
    expect(screen.getByText('Saved 5 files to disk')).toBeInTheDocument()
  })

  it('calls onDismiss via close button', async () => {
    const onDismiss = vi.fn()
    const user = userEvent.setup()
    render(<ToastContainer toasts={[{ id: '1', message: 'Close me', type: 'info' }]} onDismiss={onDismiss} />)
    await user.click(screen.getByLabelText('Dismiss notification: Close me'))
    await waitFor(() => expect(onDismiss).toHaveBeenCalledWith('1'))
  })

  it('renders Clear all when more than 2 toasts', () => {
    render(
      <ToastContainer
        toasts={[
          { id: '1', message: 'A', type: 'success' },
          { id: '2', message: 'B', type: 'info' },
          { id: '3', message: 'C', type: 'error' },
        ]}
        onDismiss={() => {}}
        onClearAll={() => {}}
      />
    )
    expect(screen.getByText('Clear all')).toBeInTheDocument()
  })

  it('hides Clear all when 2 or fewer toasts', () => {
    render(
      <ToastContainer
        toasts={[
          { id: '1', message: 'A', type: 'success' },
          { id: '2', message: 'B', type: 'info' },
        ]}
        onDismiss={() => {}}
        onClearAll={() => {}}
      />
    )
    expect(screen.queryByText('Clear all')).not.toBeInTheDocument()
  })

  it('calls onClearAll when Clear all clicked', async () => {
    const onClearAll = vi.fn()
    const user = userEvent.setup()
    render(
      <ToastContainer
        toasts={[
          { id: '1', message: 'A', type: 'success' },
          { id: '2', message: 'B', type: 'info' },
          { id: '3', message: 'C', type: 'error' },
        ]}
        onDismiss={() => {}}
        onClearAll={onClearAll}
      />
    )
    await user.click(screen.getByText('Clear all'))
    expect(onClearAll).toHaveBeenCalledOnce()
  })

  it('has accessible notification region', () => {
    render(<ToastContainer toasts={[{ id: '1', message: 'Hi', type: 'info' }]} onDismiss={() => {}} />)
    expect(screen.getByRole('region')).toHaveAttribute('aria-label', 'Notifications')
  })
})
