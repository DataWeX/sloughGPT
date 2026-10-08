// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

import { ResultsStep } from './ResultsStep'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    recover: vi.fn().mockResolvedValue({
      status: 'recovering',
      recovery_job_id: 'recovery_j4',
      original_job_id: 'j4',
      message: 'Resumed recovery_j4 from checkpoint',
    }),
  },
}))

const mockPush = vi.fn()
vi.mock('@/vite/next-compat/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))

import { trainingJobsController } from '@/lib/training-controller'

const emptyCheckpoints: UseTrainingCheckpointsReturn = {
  checkpoints: [],
  loadingCheckpoints: false,
  activeCheckpoint: null,
  builds: [],
  loadingBuilds: false,
  jobs: [],
  loadingJobs: false,
  setActiveCheckpoint: vi.fn(),
  setCheckpoints: vi.fn(),
  fetchCheckpoints: vi.fn(),
  fetchBuilds: vi.fn(),
  fetchJobs: vi.fn(),
  handleLoadCheckpoint: vi.fn(),
  handleDeleteCheckpoint: vi.fn(),
}

const checkpointsWithData: UseTrainingCheckpointsReturn = {
  ...emptyCheckpoints,
  checkpoints: [
    {
      name: 'cp-1',
      path: 'models/checkpoints/cp-1.soul',
      soul: 'soul-1',
      loss: 0.45,
      tags: ['distill'],
    },
    { name: 'cp-2', path: 'models/checkpoints/cp-2.soul', soul: 'soul-2', loss: 0.32, tags: [] },
  ],
}

const checkpointsWithJobs: UseTrainingCheckpointsReturn = {
  ...checkpointsWithData,
  jobs: [
    {
      id: 'j1',
      name: 'job-a',
      status: 'running',
      progress: 45,
      created_at: '2026-01-01T00:00:00Z',
      method: 'distill',
      dataset: 'shakespeare',
    },
    {
      id: 'j2',
      name: 'job-b',
      status: 'completed',
      progress: 100,
      created_at: '2026-01-02T00:00:00Z',
    },
    { id: 'j3', name: 'job-c', status: 'failed', progress: 0, created_at: '2026-01-03T00:00:00Z' },
  ],
}

const checkpointsWithInterrupted: UseTrainingCheckpointsReturn = {
  ...emptyCheckpoints,
  jobs: [
    {
      id: 'j4',
      name: 'job-d',
      status: 'interrupted',
      progress: 42,
      created_at: '2026-01-04T00:00:00Z',
    },
  ],
}

const checkpointsWithTurbo: UseTrainingCheckpointsReturn = {
  ...emptyCheckpoints,
  checkpoints: [
    {
      name: 'cp-1',
      path: 'models/checkpoints/cp-1.soul',
      soul: 'soul-1',
      loss: 0.45,
      tags: ['distill'],
    },
    {
      name: 'turbo-1',
      path: 'models/turbo-trained/turbo-1.soul',
      soul: 'soul-1',
      loss: 1.2,
      tags: [],
      source: 'turbo',
    },
  ],
}

const checkpointsWithQuality: UseTrainingCheckpointsReturn = {
  ...emptyCheckpoints,
  checkpoints: [
    {
      name: 'cp-1',
      path: 'models/checkpoints/cp-1.soul',
      soul: 'soul-1',
      loss: 0.45,
      avg_quality: 4.2,
      tags: ['distill'],
    },
    {
      name: 'cp-2',
      path: 'models/checkpoints/cp-2.soul',
      soul: 'soul-2',
      loss: 0.32,
      avg_quality: 3.8,
      tags: [],
    },
  ],
}

const renderStep = (checkpoints: UseTrainingCheckpointsReturn, addToast = vi.fn()) =>
  render(
    <ResultsStep
      checkpoints={checkpoints}
      goToTrain={vi.fn()}
      onTest={vi.fn()}
      addToast={addToast}
    />,
  )

describe('ResultsStep', () => {
  afterEach(cleanup)

  it('renders the step title', () => {
    renderStep(emptyCheckpoints)
    expect(screen.getByText(/Results/)).toBeDefined()
  })

  it('shows empty state when no checkpoints', () => {
    renderStep(emptyCheckpoints)
    expect(screen.getByText(/No checkpoints yet/)).toBeDefined()
  })

  it('shows checkpoint count', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('2 checkpoint(s) saved')).toBeDefined()
  })

  it('renders checkpoint names', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('cp-1')).toBeDefined()
    expect(screen.getByText('cp-2')).toBeDefined()
  })

  it('displays loss values', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('Loss: 0.4500')).toBeDefined()
    expect(screen.getByText('Loss: 0.3200')).toBeDefined()
  })

  it('displays quality values when present', () => {
    renderStep(checkpointsWithQuality)
    expect(screen.getByText('Quality: 4.2/5')).toBeDefined()
    expect(screen.getByText('Quality: 3.8/5')).toBeDefined()
  })

  it('hides quality when not present', () => {
    renderStep(checkpointsWithData)
    expect(screen.queryByText(/Quality:/)).toBeNull()
  })

  it('shows Test model button when checkpoints exist', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('Test model')).toBeDefined()
  })

  it('hides Test model button when no checkpoints', () => {
    renderStep(emptyCheckpoints)
    expect(screen.queryByText('Test model')).toBeNull()
  })

  it('offers Try it now when there is a checkpoint to load', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('Try it now')).toBeDefined()
  })

  it('hides Try it now when there is nothing to load', () => {
    renderStep(emptyCheckpoints)
    expect(screen.queryByText('Try it now')).toBeNull()
  })

  it('loads the best checkpoint, then jumps to chat', async () => {
    mockPush.mockClear()
    const cps = { ...checkpointsWithData, handleLoadCheckpoint: vi.fn().mockResolvedValue(true) }
    const addToast = vi.fn()

    renderStep(cps, addToast)
    screen.getByText('Try it now').click()

    await vi.waitFor(() => {
      expect(cps.handleLoadCheckpoint).toHaveBeenCalledWith('cp-2', addToast)
      expect(mockPush).toHaveBeenCalledWith('/chat')
    })
  })

  it('stays on the results page when the load fails', async () => {
    mockPush.mockClear()
    const cps = { ...checkpointsWithData, handleLoadCheckpoint: vi.fn().mockResolvedValue(false) }

    renderStep(cps)
    screen.getByText('Try it now').click()

    await vi.waitFor(() => {
      expect(cps.handleLoadCheckpoint).toHaveBeenCalledWith('cp-2', expect.any(Function))
    })
    expect(mockPush).not.toHaveBeenCalled()
  })

  it('offers Compare when there are at least two checkpoints', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('Compare')).toBeDefined()
  })

  it('hides Compare when a single checkpoint has nothing to compare against', () => {
    renderStep({
      ...emptyCheckpoints,
      checkpoints: [
        { name: 'solo', path: 'models/checkpoints/solo.soul', soul: 'soul-1', loss: 0.5, tags: [] },
      ],
    })
    expect(screen.queryByText('Compare')).toBeNull()
  })

  it('opens the compare dialog', () => {
    renderStep(checkpointsWithData)
    fireEvent.click(screen.getByText('Compare'))
    expect(screen.getByText('Compare checkpoints')).toBeDefined()
    expect(screen.getByLabelText('Checkpoint A')).toBeDefined()
  })

  it('has Train more button', () => {
    renderStep(emptyCheckpoints)
    expect(screen.getByText('Train more')).toBeDefined()
  })

  it('marks the lowest-loss checkpoint as Best', () => {
    renderStep(checkpointsWithData)
    const bestBadges = screen.getAllByText('Best')
    expect(bestBadges.length).toBe(1)
    expect(screen.getByText('cp-2').parentElement?.textContent).toContain('Best')
    expect(screen.getByText('cp-1').parentElement?.textContent).not.toContain('Best')
  })

  it('shows a Turbo badge only for turbo-sourced checkpoints', () => {
    renderStep(checkpointsWithTurbo)
    expect(screen.getAllByText('Turbo').length).toBe(1)
    expect(screen.getByText('turbo-1').parentElement?.textContent).toContain('Turbo')
    expect(screen.getByText('cp-1').parentElement?.textContent).not.toContain('Turbo')
  })

  it('renders a Delete button per checkpoint and calls handleDeleteCheckpoint', () => {
    const handleDeleteCheckpoint = vi.fn()
    renderStep({ ...checkpointsWithData, handleDeleteCheckpoint })
    const deleteButtons = screen.getAllByText('Delete')
    expect(deleteButtons.length).toBe(2)
    vi.stubGlobal(
      'confirm',
      vi.fn(() => true),
    )
    deleteButtons[0].click()
    expect(handleDeleteCheckpoint).toHaveBeenCalledWith(
      'cp-1',
      expect.any(Function),
      'models/checkpoints/cp-1.soul',
    )
    vi.unstubAllGlobals()
  })

  it('shows no Recent runs section when there are no finished jobs', () => {
    renderStep(checkpointsWithData)
    expect(screen.queryByText('Recent runs')).toBeNull()
  })

  it('renders recent finished jobs with status badges', () => {
    renderStep(checkpointsWithJobs)
    // Only completed/failed jobs are listed — in-flight runs are hidden.
    expect(screen.queryByText('job-a')).toBeNull()
    expect(screen.getByText('job-b')).toBeDefined()
    expect(screen.getByText('job-c')).toBeDefined()
    expect(screen.getByText('Completed')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })

  it('hides in-flight jobs instead of showing progress', () => {
    renderStep(checkpointsWithJobs)
    expect(screen.queryByText('45%')).toBeNull()
    expect(screen.queryByText('job-a')).toBeNull()
  })

  it('passes addToast to handleLoadCheckpoint', () => {
    const handleLoadCheckpoint = vi.fn()
    const addToast = vi.fn()
    renderStep({ ...checkpointsWithData, handleLoadCheckpoint }, addToast)
    screen.getAllByText('Load')[0].click()
    expect(handleLoadCheckpoint).toHaveBeenCalledWith('cp-1', addToast)
  })

  it('includes interrupted jobs in Recent runs with Interrupted badge', () => {
    renderStep(checkpointsWithInterrupted)
    expect(screen.getByText('Recent runs')).toBeDefined()
    expect(screen.getByText('job-d')).toBeDefined()
    expect(screen.getByText('Interrupted')).toBeDefined()
  })

  it('shows Resume for interrupted jobs and calls recover + onRecovered', async () => {
    const addToast = vi.fn()
    const onRecovered = vi.fn()
    render(
      <ResultsStep
        checkpoints={checkpointsWithInterrupted}
        goToTrain={vi.fn()}
        onTest={vi.fn()}
        addToast={addToast}
        onRecovered={onRecovered}
      />,
    )
    const resume = screen.getByRole('button', { name: 'Resume' })
    resume.click()
    await vi.waitFor(() => {
      expect(trainingJobsController.recover).toHaveBeenCalledWith('j4')
      expect(onRecovered).toHaveBeenCalled()
    })
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('recovery_j4'), 'success')
  })

  it('surfaces the backend reason when resume fails', async () => {
    const addToast = vi.fn()
    vi.mocked(trainingJobsController.recover).mockRejectedValueOnce(
      new Error('No dataset recorded for this job.'),
    )
    render(
      <ResultsStep
        checkpoints={checkpointsWithInterrupted}
        goToTrain={vi.fn()}
        onTest={vi.fn()}
        addToast={addToast}
        onRecovered={vi.fn()}
      />,
    )
    screen.getByRole('button', { name: 'Resume' }).click()
    await vi.waitFor(() => {
      expect(addToast).toHaveBeenCalledWith(
        'Could not resume job: No dataset recorded for this job.',
        'error',
      )
    })
  })
})
