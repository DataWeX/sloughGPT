// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    recoverable: vi.fn(),
    recover: vi.fn(),
    abandon: vi.fn(),
  },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,
  AlertDialog: ({ open, children }: any) => open ? <div role="dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: any) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
  AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
}))

import { RecoveryCard } from './RecoveryCard'
import { trainingJobsController } from '@/lib/training-controller'

const mockToast = vi.fn()
const jobs = [
  { id: 'j1', type: 'auto-train', error: 'OOM', created_at: '2025-01-01T00:00:00Z' },
  { id: 'j2', type: 'lora', error: 'Timeout', created_at: '2025-01-02T00:00:00Z' },
]

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.recoverable).mockResolvedValue(jobs as any)
  vi.mocked(trainingJobsController.recover).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.abandon).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('RecoveryCard', () => {
  it('returns null when no recoverable jobs', async () => {
    vi.mocked(trainingJobsController.recoverable).mockResolvedValue([])
    const { container } = render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => {
      expect(container.querySelector('[data-testid="skeleton"]')).toBeNull()
    })
    expect(container.innerHTML).toBe('')
  })

  it('shows loading skeletons', () => {
    vi.mocked(trainingJobsController.recoverable).mockReturnValue(new Promise(() => {}))
    render(<RecoveryCard addToast={mockToast} />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
  })

  it('loads and displays recoverable jobs', async () => {
    render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Recoverable Jobs (2)')).toBeDefined()
    })
  })

  it('shows Recover and Abandon buttons per job', async () => {
    render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getAllByText('Recover').length).toBe(2)
      expect(screen.getAllByText('Abandon').length).toBe(2)
    })
  })

  it('recovers a job', async () => {
    render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getAllByText('Recover').length).toBe(2) })
    const recovers = screen.getAllByText('Recover')
    fireEvent.click(recovers[0])
    await waitFor(() => {
      expect(trainingJobsController.recover).toHaveBeenCalledWith('j1')
    })
  })

  it('opens abandon confirmation dialog', async () => {
    render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getAllByText('Abandon').length).toBe(2) })
    const abandons = screen.getAllByText('Abandon')
    fireEvent.click(abandons[0])
    expect(screen.getByText('Abandon this job?')).toBeDefined()
  })

  it('abandons job on confirm', async () => {
    render(<RecoveryCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getAllByText('Abandon').length).toBe(2) })
    const abandons = screen.getAllByText('Abandon')
    fireEvent.click(abandons[0])
    fireEvent.click(screen.getByText('Abandon Job'))
    await waitFor(() => {
      expect(trainingJobsController.abandon).toHaveBeenCalledWith('j1')
    })
  })
})
