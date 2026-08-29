// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    trainFromFeedback: vi.fn(),
    get: vi.fn(),
    stop: vi.fn(),
  },
}))
vi.mock('@/lib/souls-controller', () => ({
  soulsController: { loadCheckpoint: vi.fn() },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Input: (p: any) => <input {...p} />,
  Label: ({ children }: any) => <label>{children}</label>,
  Progress: (p: any) => <div data-testid="progress" data-value={p.value} />,
  AlertDialog: ({ open, children }: any) => open ? <div role="dialog">{children}</div> : null,
  AlertDialogContent: ({ children }: any) => <div>{children}</div>,
  AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
  AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
  AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
  AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
  AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
}))

import { FeedbackTrainCard } from './FeedbackTrainCard'
import { trainingJobsController } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'

const mockToast = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.trainFromFeedback).mockResolvedValue({ job_id: 'j1', samples: 10 } as any)
  vi.mocked(trainingJobsController.get).mockResolvedValue(null as any)
  vi.mocked(trainingJobsController.stop).mockResolvedValue(undefined as any)
  vi.mocked(soulsController.loadCheckpoint).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('FeedbackTrainCard', () => {
  it('renders with title and train button', () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    expect(screen.getAllByText('Train from feedback').length).toBeGreaterThanOrEqual(1)
  })

  it('toggles config panel', () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    fireEvent.click(screen.getByText('Show config'))
    expect(screen.getByText('Epochs')).toBeDefined()
    expect(screen.getByText('Learning Rate')).toBeDefined()
    expect(screen.getByText('Batch Size')).toBeDefined()
  })

  it('starts training on button click', async () => {
    render(<FeedbackTrainCard addToast={mockToast} />)
    const btns = screen.getAllByText('Train from feedback')
    fireEvent.click(btns[btns.length - 1])
    await waitFor(() => {
      expect(trainingJobsController.trainFromFeedback).toHaveBeenCalled()
    })
  })
})
