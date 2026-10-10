// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets } from './__test-helper'

vi.mock('@/components/training/TrainingPresets', () => ({
  TrainingPresets: () => <div data-testid="training-presets" />,
}))

// jsdom does not toggle native <details> on summary click, so FoldSection gets
// a stateful stand-in: children render only while open (same contract).
vi.mock('@sloughgpt/strui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@sloughgpt/strui')>()
  return {
    ...actual,
    FoldSection: ({ heading, children }: { heading: React.ReactNode; children: React.ReactNode }) => {
      const [open, setOpen] = React.useState(false)
      return (
        <details open={open}>
          <summary onClick={() => setOpen((o) => !o)}>{heading}</summary>
          {open && <div>{children}</div>}
        </details>
      )
    },
  }
})

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

describe('ConfigureStep advanced options fold', () => {
  afterEach(cleanup)

  it('hides expert controls by default', () => {
    renderStep(baseForm)
    expect(screen.queryByLabelText('Epochs')).toBeNull()
    expect(screen.queryByLabelText('Batch size')).toBeNull()
    expect(screen.queryByLabelText('Learning rate')).toBeNull()
  })

  it('reveals expert controls when advanced options are opened', () => {
    renderStep(baseForm)
    fireEvent.click(screen.getByText('Show advanced options'))
    expect(screen.getByLabelText('Epochs')).toBeTruthy()
    expect(screen.getByLabelText('Batch size')).toBeTruthy()
    expect(screen.getByLabelText('Learning rate')).toBeTruthy()
  })

  it('keeps the base model picker in the main flow', () => {
    renderStep({ ...baseForm, method: 'finetune', availableModels: ['gpt2'] })
    expect(screen.getByLabelText('Base model')).toBeTruthy()
  })

  it('keeps hyperparameter validation errors visible while collapsed', () => {
    renderStep({ ...baseForm, trainingEpochs: 999 })
    expect(screen.getByText('Epochs must be 1–500')).toBeTruthy()
  })
})
