// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets, makeCheckpoints } from './__test-helper'

vi.mock('@/components/training/TrainingPresets', () => ({
  TrainingPresets: () => <div data-testid="training-presets" />,
}))

// jsdom does not toggle native <details> on summary click, so FoldSection gets
// a stateful stand-in: children render only while open (same contract).
vi.mock('@sloughgpt/strui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@sloughgpt/strui')>()
  return {
    ...actual,
    // The real Select is a popover; a native <select> keeps value +
    // onValueChange observable (same contract as QuickTrainCard.test.tsx).
    Select: ({ children, value, onValueChange }: any) => (
      <select value={value} onChange={(e) => onValueChange?.((e.target as HTMLSelectElement).value)}>
        {children}
      </select>
    ),
    SelectTrigger: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    SelectValue: ({ children }: any) => <span>{children}</span>,
    // Fragment: <option> must be a direct child of <select> or select.value is ''
    SelectContent: ({ children }: any) => <>{children}</>,
    SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
    FoldSection: ({
      heading,
      children,
    }: {
      heading: React.ReactNode
      children: React.ReactNode
    }) => {
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
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

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
    renderStep({ ...baseForm, trainingEpochs: 9999 })
    expect(screen.getByText('Epochs must be 1–1000')).toBeTruthy()
  })
})

describe('ConfigureStep hyperparameter sanitization', () => {
  afterEach(cleanup)

  const openAdvanced = () => fireEvent.click(screen.getByText('Show advanced options'))

  it('flags batch size beyond the shared limits', () => {
    renderStep({ ...baseForm, trainingBatchSize: 9999 })
    expect(screen.getByText('Batch size must be 1–512')).toBeTruthy()
  })

  it('flags learning rate beyond the shared limits', () => {
    renderStep({ ...baseForm, trainingLR: 5 })
    expect(screen.getByText('Learning rate must be 0.00001–1')).toBeTruthy()
  })

  it('flags NaN hyperparameters instead of passing validation', () => {
    renderStep({ ...baseForm, trainingEpochs: Number.NaN })
    expect(screen.getByText('Epochs must be 1–1000')).toBeTruthy()
  })

  it('lets the learning rate be typed without snapping mid-edit', () => {
    // Reported bug: the old onChange (`Number(x) || 1e-3`) replaced the
    // user's draft on every keystroke, so `0.001` could never be typed —
    // typing `0` immediately snapped to `0.001` and the edit was fought.
    const form = makeForm({ trainingLR: 1 })
    renderStep(form)
    openAdvanced()
    const lr = screen.getByLabelText('Learning rate') as HTMLInputElement

    fireEvent.change(lr, { target: { value: '0' } })
    expect(lr.value).toBe('0')
    expect(form.setTrainingLR).not.toHaveBeenCalled()

    fireEvent.change(lr, { target: { value: '0.001' } })
    expect(lr.value).toBe('0.001')
    expect(form.setTrainingLR).not.toHaveBeenCalled()

    fireEvent.blur(lr)
    expect(form.setTrainingLR).toHaveBeenLastCalledWith(0.001)
    expect(lr.value).toBe('0.001')
  })

  it('clamps an out-of-range learning rate on blur', () => {
    const form = makeForm({ trainingLR: 0.001 })
    renderStep(form)
    openAdvanced()
    const lr = screen.getByLabelText('Learning rate') as HTMLInputElement

    fireEvent.change(lr, { target: { value: '999' } })
    expect(form.setTrainingLR).not.toHaveBeenCalled()
    fireEvent.blur(lr)
    expect(form.setTrainingLR).toHaveBeenLastCalledWith(1)
    expect(lr.value).toBe('1')
  })

  it('commits a clamped epochs value only on blur', () => {
    const form = makeForm({ trainingEpochs: 10 })
    renderStep(form)
    openAdvanced()
    const epochs = screen.getByLabelText('Epochs') as HTMLInputElement

    fireEvent.change(epochs, { target: { value: '5000' } })
    expect(form.setTrainingEpochs).not.toHaveBeenCalled()
    fireEvent.blur(epochs)
    expect(form.setTrainingEpochs).toHaveBeenLastCalledWith(1000)
    expect(epochs.value).toBe('1000')
  })

  it('reverts to the last valid value when the field is left empty', () => {
    const form = makeForm({ trainingEpochs: 10 })
    renderStep(form)
    openAdvanced()
    const epochs = screen.getByLabelText('Epochs') as HTMLInputElement

    fireEvent.change(epochs, { target: { value: '' } })
    fireEvent.blur(epochs)
    expect(form.setTrainingEpochs).not.toHaveBeenCalled()
    expect(epochs.value).toBe('10')
  })
})

// ---------------------------------------------------------------------------
// Resume default (card 20261119_133132): one-shot latest-of-the-same-dataset
// + a touched guard. Uses a harness that holds the value in real state — the
// makeForm() setters are spies, so they cannot show what the select renders.
// ---------------------------------------------------------------------------

describe('ConfigureStep resume default', () => {
  afterEach(cleanup)

  function ResumeHarness({
    form: initial,
    datasets: datasetsProp,
    checkpoints: checkpointsProp,
  }: {
    form: TrainingFormState
    datasets: UseTrainingDatasetsReturn
    checkpoints: UseTrainingCheckpointsReturn
  }) {
    const [resumeCheckpoint, setResumeCheckpoint] = React.useState(initial.resumeCheckpoint)
    const form = { ...initial, resumeCheckpoint, setResumeCheckpoint }
    return (
      <ConfigureStep
        form={form}
        datasets={datasetsProp}
        checkpoints={checkpointsProp}
        onNext={vi.fn()}
        onBack={vi.fn()}
      />
    )
  }

  // The base-model Select needs method=finetune + availableModels, so in this
  // configuration the resume select is the only <select> on screen.
  const resumeSelect = (container: HTMLElement) =>
    container.querySelector('select') as HTMLSelectElement

  const openAdvanced = () => fireEvent.click(screen.getByText('Show advanced options'))

  const shakespeareCkpts = [
    { name: 'shakespeare_1700000100.soul', soul: 'shakespeare', created_at: '2026-11-01T00:00:00Z' },
    { name: 'other_1700000999.soul', soul: 'other', created_at: '2026-11-09T00:00:00Z' },
    { name: 'shakespeare_1700000200.soul', soul: 'shakespeare', created_at: '2026-11-05T00:00:00Z' },
    // training_dataset is a PATH while selectedDataset is an id: the match is
    // the path component, not a raw === (train_pipeline writes data_path).
    {
      name: 'run_555.soul',
      soul: 'run',
      training_dataset: 'datasets/shakespeare/input.txt',
      created_at: '2026-11-08T00:00:00Z',
    },
  ]

  it('defaults to the latest checkpoint trained on the selected dataset', () => {
    const { container } = render(
      <ResumeHarness
        form={makeForm()}
        datasets={makeDatasets({ selectedDataset: 'shakespeare' })}
        checkpoints={makeCheckpoints({ checkpoints: shakespeareCkpts })}
      />,
    )
    openAdvanced()

    // latest by created_at among the three same-dataset matches: run_555
    // (2026-11-08) beats the two shakespeare_* filenames; other_1700000999 is
    // newer still but belongs to another dataset.
    expect(resumeSelect(container).value).toBe('run_555.soul')
  })

  it('stays on Start fresh when nothing was trained on the selected dataset', () => {
    const { container } = render(
      <ResumeHarness
        form={makeForm()}
        datasets={makeDatasets({ selectedDataset: 'shakespeare' })}
        checkpoints={makeCheckpoints({
          checkpoints: [
            { name: 'other_1700000999.soul', soul: 'other' },
            // field present and authoritative: the filename says shakespeare
            // but the recorded corpus is another dataset -> not a match.
            {
              name: 'shakespeare_1700000300.soul',
              soul: 'shakespeare',
              training_dataset: 'datasets/other/input.txt',
            },
          ],
        })}
      />,
    )
    openAdvanced()

    expect(resumeSelect(container).value).toBe('')
  })

  it('never steals the choice back after the user picks, including Start fresh', () => {
    const datasetsProp = makeDatasets({ selectedDataset: 'shakespeare' })
    const view = render(
      <ResumeHarness
        form={makeForm()}
        datasets={datasetsProp}
        checkpoints={makeCheckpoints({ checkpoints: shakespeareCkpts })}
      />,
    )
    openAdvanced()
    const select = resumeSelect(view.container)
    expect(select.value).toBe('run_555.soul')

    // user deliberately picks "Start fresh"
    fireEvent.change(select, { target: { value: '' } })
    expect(select.value).toBe('')

    // a newer same-dataset checkpoint lands afterwards (fetch race): the
    // one-shot default must not overwrite an explicit choice.
    view.rerender(
      <ResumeHarness
        form={makeForm()}
        datasets={datasetsProp}
        checkpoints={makeCheckpoints({
          checkpoints: [
            ...shakespeareCkpts,
            {
              name: 'run_999.soul',
              soul: 'run',
              training_dataset: 'datasets/shakespeare/input.txt',
              created_at: '2026-11-12T00:00:00Z',
            },
          ],
        })}
      />,
    )
    expect(resumeSelect(view.container).value).toBe('')
  })
})
