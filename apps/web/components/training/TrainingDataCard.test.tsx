// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    getTrainingStats: vi.fn(),
    listTrainingPairs: vi.fn(),
    deletePair: vi.fn(),
    updatePairQuality: vi.fn(),
    deleteSyncedPairs: vi.fn(),
  },
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Input: (p: any) => <input {...p} />,
}))

import { TrainingDataCard } from './TrainingDataCard'
import { trainingJobsController } from '@/lib/training-controller'

const mockToast = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.getTrainingStats).mockResolvedValue({ total: 100, pending: 10, synced: 50, used: 40 } as any)
  vi.mocked(trainingJobsController.listTrainingPairs).mockResolvedValue({ pairs: [], total: 0 } as any)
  vi.mocked(trainingJobsController.deletePair).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.updatePairQuality).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.deleteSyncedPairs).mockResolvedValue(undefined as any)
})

afterEach(() => cleanup())

describe('TrainingDataCard', () => {
  it('displays stats', async () => {
    render(<TrainingDataCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('100')).toBeDefined()
      expect(screen.getByText('10')).toBeDefined()
      expect(screen.getByText('50')).toBeDefined()
    })
  })

  it('calls APIs on mount', async () => {
    render(<TrainingDataCard addToast={mockToast} />)
    await waitFor(() => {
      expect(trainingJobsController.listTrainingPairs).toHaveBeenCalled()
      expect(trainingJobsController.getTrainingStats).toHaveBeenCalled()
    })
  })
})
