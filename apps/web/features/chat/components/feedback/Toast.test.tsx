/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ToastContainer, RadixToastContainer } from './Toast'

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

// AppLayout renders RadixToastContainer fed by useToastStore. Its toasts must
// portal to document.body: `.sl-app-shell` is height:100dvh + overflow:hidden,
// and the toast variant is `relative`, so inline rendering drops them into the
// shell's flex column where overflow:hidden clips them off-screen (the
// "clipped Loaded: toast" defect).
describe('RadixToastContainer', () => {
  it('leaves the host container empty (toasts are portaled, not inline)', () => {
    const { container } = render(
      <RadixToastContainer
        toasts={[{ id: '1', message: 'Loaded: my session', type: 'success' }]}
        onDismiss={() => {}}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders store toasts into a fixed Notifications region in document.body', () => {
    render(
      <RadixToastContainer
        toasts={[{ id: '1', message: 'Loaded: my session', type: 'success' }]}
        onDismiss={() => {}}
      />
    )
    const region = document.body.querySelector('[role="region"][aria-label="Notifications"]')
    expect(region).not.toBeNull()
    expect(region?.className).toContain('fixed')
    expect(screen.getByText('Loaded: my session')).toBeInTheDocument()
  })

  it('renders verbose + dismiss controls for store toasts', async () => {
    const onDismiss = vi.fn()
    const user = userEvent.setup()
    render(
      <RadixToastContainer
        toasts={[{ id: '9', message: 'Saved', type: 'info', verbose: 'Saved 5 files to disk' }]}
        onDismiss={onDismiss}
      />
    )
    expect(screen.getByText('Saved 5 files to disk')).toBeInTheDocument()
    await user.click(screen.getByLabelText('Close'))
    expect(onDismiss).toHaveBeenCalledWith('9')
  })

  it('renders Clear all only when more than 2 store toasts', () => {
    const three = [
      { id: '1', message: 'A', type: 'success' as const },
      { id: '2', message: 'B', type: 'info' as const },
      { id: '3', message: 'C', type: 'error' as const },
    ]
    const { unmount } = render(
      <RadixToastContainer toasts={three} onDismiss={() => {}} onClearAll={() => {}} />
    )
    expect(screen.getByText('Clear all')).toBeInTheDocument()
    unmount()
    cleanup()
    render(
      <RadixToastContainer toasts={three.slice(0, 2)} onDismiss={() => {}} onClearAll={() => {}} />
    )
    expect(screen.queryByText('Clear all')).not.toBeInTheDocument()
  })
})
