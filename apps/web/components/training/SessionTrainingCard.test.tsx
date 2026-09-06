// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

import { SessionTrainingCard } from './SessionTrainingCard'

const mockAddToast = vi.fn()

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: mockAddToast }),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listChatSessions: vi.fn(),
    getSessionPairs: vi.fn(),
    trainFromSessions: vi.fn(),
  },
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
    Checkbox: ({ checked, onCheckedChange, className, ...props }: any) => (
      <input type="checkbox" checked={checked} onChange={() => onCheckedChange?.(!checked)} className={className} {...props} />
    ),
  }
})
vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, title }: any) => open ? (
    <div role="dialog">
      <p>{title}</p>
      <button onClick={onConfirm}>Confirm</button>
    </div>
  ) : null,
}))

import { trainingJobsController } from '@/lib/training-controller'

describe('SessionTrainingCard', () => {
  afterEach(cleanup)
  beforeEach(() => { vi.clearAllMocks() })

  it('renders loading state', () => {
    vi.mocked(trainingJobsController.listChatSessions).mockReturnValue(new Promise(() => {}))
    render(<SessionTrainingCard addToast={mockAddToast} />)
    expect(screen.getByText('Loading sessions...')).toBeDefined()
  })

  it('renders sessions list', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockResolvedValue([
      { id: 's1', name: 'Chat 1', updated_at: '2025-01-01T00:00:00Z' },
      { id: 's2', name: 'Chat 2', updated_at: '2025-01-02T00:00:00Z' },
    ])
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => {
      expect(screen.getByText('Chat 1')).toBeDefined()
      expect(screen.getByText('Chat 2')).toBeDefined()
    })
  })

  it('shows empty state', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockResolvedValue([])
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => {
      expect(screen.getByText('No chat sessions found.')).toBeDefined()
    })
  })

  it('selects and trains from sessions', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockResolvedValue([
      { id: 's1', name: 'Chat 1', updated_at: '2025-01-01T00:00:00Z' },
    ])
    vi.mocked(trainingJobsController.getSessionPairs).mockResolvedValue({ pairs: [], count: 5 })
    vi.mocked(trainingJobsController.trainFromSessions).mockResolvedValue({
      success: true, checkpoint_name: 'cp', loss: 0.42, steps: 100, elapsed_ms: 1000,
    })
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => { expect(screen.getByText('Chat 1')).toBeDefined() })
    fireEvent.click(screen.getByText('Select all'))
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Train from Sessions' })).toBeDefined()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Train from Sessions' }))
    await waitFor(() => { expect(screen.getByRole('dialog')).toBeDefined() })
    fireEvent.click(screen.getByText('Confirm'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Trained from 1 sessions'), 'success'
      )
    })
  })

  it('select all / deselect all', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockResolvedValue([
      { id: 's1', name: 'Chat 1', updated_at: '2025-01-01T00:00:00Z' },
      { id: 's2', name: 'Chat 2', updated_at: '2025-01-02T00:00:00Z' },
    ])
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => { expect(screen.getByText('Chat 1')).toBeDefined() })
    fireEvent.click(screen.getByText('Select all'))
    await waitFor(() => { expect(screen.getByText('Deselect all')).toBeDefined() })
    fireEvent.click(screen.getByText('Deselect all'))
    await waitFor(() => { expect(screen.getByText('Select all')).toBeDefined() })
  })

  it('handles load error', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockRejectedValue(new Error('Network error'))
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load chat sessions', 'error')
    })
  })

  it('handles training failure', async () => {
    vi.mocked(trainingJobsController.listChatSessions).mockResolvedValue([
      { id: 's1', name: 'Chat 1', updated_at: '2025-01-01T00:00:00Z' },
    ])
    vi.mocked(trainingJobsController.getSessionPairs).mockResolvedValue({ pairs: [], count: 5 })
    vi.mocked(trainingJobsController.trainFromSessions).mockRejectedValue(new Error('Train failed'))
    render(<SessionTrainingCard addToast={mockAddToast} />)
    await waitFor(() => { expect(screen.getByText('Chat 1')).toBeDefined() })
    fireEvent.click(screen.getByText('Select all'))
    await waitFor(() => { expect(screen.getByRole('button', { name: 'Train from Sessions' })).toBeDefined() })
    fireEvent.click(screen.getByRole('button', { name: 'Train from Sessions' }))
    await waitFor(() => { expect(screen.getByRole('dialog')).toBeDefined() })
    fireEvent.click(screen.getByText('Confirm'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith(
        expect.stringContaining('Session training failed'), 'error'
      )
    })
  })
})
