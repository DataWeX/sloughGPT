// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listBuilds: vi.fn(),
    deleteCheckpoint: vi.fn(),
    downloadCheckpoint: vi.fn(),
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
}))

import { TrainingBuildsCard } from './TrainingBuildsCard'
import { trainingJobsController } from '@/lib/training-controller'
import { soulsController } from '@/lib/souls-controller'

const mockToast = vi.fn()
const builds = [
  { name: 'build-1', build_type: 'auto-train', loss: 0.3, epochs: 5, size_mb: 10, model: 'gpt2' },
  { name: 'build-2', build_type: 'lora', loss: 0.2, epochs: 3, size_mb: 5, model: 'qwen' },
  { name: 'build-3', build_type: 'auto-train', loss: 0.1, epochs: 10, size_mb: 20, model: 'gpt2' },
]

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(trainingJobsController.listBuilds).mockResolvedValue(builds as any)
  vi.mocked(soulsController.loadCheckpoint).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.deleteCheckpoint).mockResolvedValue(undefined as any)
  vi.mocked(trainingJobsController.downloadCheckpoint).mockResolvedValue(new Blob() as any)
})

afterEach(() => cleanup())

describe('TrainingBuildsCard', () => {
  it('loads and displays builds with title', async () => {
    render(<TrainingBuildsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('Builds (3)')).toBeDefined()
      expect(screen.getByText('build-1')).toBeDefined()
      expect(screen.getByText('build-2')).toBeDefined()
      expect(screen.getByText('build-3')).toBeDefined()
    })
  })

  it('filters builds by type', async () => {
    render(<TrainingBuildsCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('build-1')).toBeDefined() })
    const loraBtn = screen.getByText('lora (1)')
    fireEvent.click(loraBtn)
    expect(screen.queryByText('build-1')).toBeNull()
    expect(screen.getByText('build-2')).toBeDefined()
  })

  it('loads checkpoint on Load click', async () => {
    render(<TrainingBuildsCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('build-1')).toBeDefined() })
    const loads = screen.getAllByText('Load')
    fireEvent.click(loads[0])
    await waitFor(() => {
      expect(soulsController.loadCheckpoint).toHaveBeenCalledWith('build-1')
    })
  })

  it('deletes checkpoint on Delete click', async () => {
    render(<TrainingBuildsCard addToast={mockToast} />)
    await waitFor(() => { expect(screen.getByText('build-1')).toBeDefined() })
    const deletes = screen.getAllByText('Delete')
    fireEvent.click(deletes[0])
    await waitFor(() => {
      expect(trainingJobsController.deleteCheckpoint).toHaveBeenCalledWith('build-1')
    })
  })

  it('shows empty state when no builds', async () => {
    vi.mocked(trainingJobsController.listBuilds).mockResolvedValue([])
    render(<TrainingBuildsCard addToast={mockToast} />)
    await waitFor(() => {
      expect(screen.getByText('No builds found. Start training to create builds.')).toBeDefined()
    })
  })
})
