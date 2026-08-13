// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import { ResultsStep } from './ResultsStep'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

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
    { name: 'cp-1', soul: 'soul-1', loss: 0.45, tags: ['distill'] },
    { name: 'cp-2', soul: 'soul-2', loss: 0.32, tags: [] },
  ],
}

const checkpointsWithJobs: UseTrainingCheckpointsReturn = {
  ...checkpointsWithData,
  jobs: [
    { id: 'j1', name: 'job-a', status: 'running', progress: 45, created_at: '2026-01-01T00:00:00Z', method: 'distill', dataset: 'shakespeare' },
    { id: 'j2', name: 'job-b', status: 'completed', progress: 100, created_at: '2026-01-02T00:00:00Z' },
    { id: 'j3', name: 'job-c', status: 'failed', progress: 0, created_at: '2026-01-03T00:00:00Z' },
  ],
}

const renderStep = (checkpoints: UseTrainingCheckpointsReturn, addToast = vi.fn()) =>
  render(<ResultsStep checkpoints={checkpoints} goToTrain={vi.fn()} onTest={vi.fn()} addToast={addToast} />)

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

  it('shows Test model button when checkpoints exist', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText('Test model')).toBeDefined()
  })

  it('hides Test model button when no checkpoints', () => {
    renderStep(emptyCheckpoints)
    expect(screen.queryByText('Test model')).toBeNull()
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

  it('renders a Delete button per checkpoint and calls handleDeleteCheckpoint', () => {
    const handleDeleteCheckpoint = vi.fn()
    renderStep({ ...checkpointsWithData, handleDeleteCheckpoint })
    const deleteButtons = screen.getAllByText('Delete')
    expect(deleteButtons.length).toBe(2)
    vi.stubGlobal('confirm', vi.fn(() => true))
    deleteButtons[0].click()
    expect(handleDeleteCheckpoint).toHaveBeenCalledWith('cp-1', expect.any(Function))
    vi.unstubAllGlobals()
  })

  it('shows jobs empty state when no jobs', () => {
    renderStep(checkpointsWithData)
    expect(screen.getByText(/No jobs yet/)).toBeDefined()
  })

  it('renders recent training jobs with status badges', () => {
    renderStep(checkpointsWithJobs)
    expect(screen.getByText('job-a')).toBeDefined()
    expect(screen.getByText('job-b')).toBeDefined()
    expect(screen.getByText('job-c')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
    expect(screen.getByText('Completed')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })

  it('shows progress percentage for in-flight jobs only', () => {
    renderStep(checkpointsWithJobs)
    expect(screen.getByText('45%')).toBeDefined()
    expect(screen.queryByText('100%')).toBeNull()
  })

  it('passes addToast to handleLoadCheckpoint', () => {
    const handleLoadCheckpoint = vi.fn()
    const addToast = vi.fn()
    renderStep({ ...checkpointsWithData, handleLoadCheckpoint }, addToast)
    screen.getAllByText('Load')[0].click()
    expect(handleLoadCheckpoint).toHaveBeenCalledWith('cp-1', addToast)
  })
})
