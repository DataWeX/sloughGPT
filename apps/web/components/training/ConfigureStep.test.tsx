// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets } from './__test-helper'

vi.mock('@/components/training/TrainingPresets', () => ({
  TrainingPresets: () => <div data-testid="training-presets" />,
}))

import { ConfigureStep } from './ConfigureStep'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

const datasets: UseTrainingDatasetsReturn = makeDatasets()

const baseForm: TrainingFormState = makeForm({
  trainingEpochs: 10,
  trainingBatchSize: 32,
  nativeEmbed: 128,
  nativeLayers: 2,
})

const renderStep = (form: TrainingFormState) =>
  render(<ConfigureStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)

const checkpoints = {
  jobs: [],
  checkpoints: [
    { name: 'cp-a.soul', path: 'models/auto-training/cp-a.soul', loss: 0.5 },
    { name: 'twin.soul', path: 'data/user_adapters/twin.soul', loss: 0.25 },
  ],
  fetchCheckpoints: vi.fn(),
  fetchJobs: vi.fn(),
} as any

const renderWithCheckpoints = (form: TrainingFormState) =>
  render(
    <ConfigureStep
      form={form}
      datasets={datasets}
      checkpoints={checkpoints}
      onNext={vi.fn()}
      onBack={vi.fn()}
    />,
  )

describe('ConfigureStep', () => {
  afterEach(cleanup)

  it('shows the resume select for methods whose start request accepts a checkpoint', () => {
    renderWithCheckpoints(baseForm)
    expect(screen.getByText('Resume from checkpoint (optional)')).toBeDefined()
  })

  it('shows the resume select for native training', () => {
    renderWithCheckpoints({ ...baseForm, method: 'native' })
    expect(screen.getByText('Resume from checkpoint (optional)')).toBeDefined()
  })

  it('hides the resume select for finetune (HF LoRA job, no .soul resume)', () => {
    renderWithCheckpoints({ ...baseForm, method: 'finetune' })
    expect(screen.queryByText('Resume from checkpoint (optional)')).toBeNull()
  })

  it('hides the resume select for vlm (visual trainer, no checkpoint field)', () => {
    renderWithCheckpoints({ ...baseForm, method: 'vlm' })
    expect(screen.queryByText('Resume from checkpoint (optional)')).toBeNull()
  })

  it('renders the step title', () => {
    renderStep(baseForm)
    expect(screen.getByText(/Configure training/)).toBeDefined()
  })

  it('shows training presets', () => {
    renderStep(baseForm)
    expect(screen.getByTestId('training-presets')).toBeDefined()
  })

  it('shows method description for distill', () => {
    renderStep(baseForm)
    expect(screen.getByText(/Train a small model from text data/)).toBeDefined()
  })

  it('shows method description for finetune', () => {
    renderStep({ ...baseForm, method: 'finetune' })
    expect(screen.getByText(/Continue training an existing model/)).toBeDefined()
  })

  it('shows method description for native', () => {
    renderStep({ ...baseForm, method: 'native' })
    expect(screen.getByText(/pure transformer from scratch/)).toBeDefined()
  })

  it('shows Back button', () => {
    renderStep(baseForm)
    expect(screen.getByText('Back')).toBeDefined()
  })

  it('displays text input mode when selected', () => {
    renderStep({ ...baseForm, inputMode: 'text' })
    expect(screen.getByLabelText(/Training text input/)).toBeDefined()
  })

  it('hides text input when dataset mode selected', () => {
    renderStep(baseForm)
    expect(screen.queryByLabelText(/Training text input/)).toBeNull()
  })

  it('shows the quality picker for native training', () => {
    renderStep({ ...baseForm, method: 'native' })
    expect(screen.getByText('Quality')).toBeDefined()
    expect(screen.getByTestId('quality-hint').textContent).toMatch(/Custom size/)
  })

  it('hides the quality picker for distill training', () => {
    renderStep(baseForm)
    expect(screen.queryByText('Quality')).toBeNull()
    expect(screen.queryByTestId('quality-hint')).toBeNull()
  })

  it('applies the picked quality bucket', () => {
    const form = makeForm({ method: 'native', quality: 'medium' })
    renderStep(form)
    screen.getByText('High').click()
    expect(form.setQuality).toHaveBeenCalledWith('high')
  })

  it('explains what the selected quality means', () => {
    renderStep(makeForm({ method: 'native', quality: 'high' }))
    expect(screen.getByTestId('quality-hint').textContent).toMatch(/Bigger model/)
  })
})
