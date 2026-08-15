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
  algo: 'bpe',
  trainingEpochs: 10,
  trainingBatchSize: 32,
  nativeEmbed: 128,
  nativeLayers: 2,
})

const renderStep = (form: TrainingFormState) =>
  render(<ConfigureStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)

describe('ConfigureStep', () => {
  afterEach(cleanup)

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
})
