// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets, ds } from './__test-helper'

vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: (props: any) => (
    <div data-testid="dataset-selector">{props.value || 'none'}</div>
  ),
}))

import { DataStep } from './DataStep'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'

const form: TrainingFormState = makeForm({
  trainingEpochs: 10,
  trainingBatchSize: 32,
  nativeEmbed: 128,
  nativeLayers: 2,
})

const datasets: UseTrainingDatasetsReturn = makeDatasets()

describe('DataStep', () => {
  afterEach(cleanup)

  it('renders the step title', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Pick your data/)).toBeDefined()
  })

  it('renders DatasetSelector', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toBeDefined()
  })

  it('shows Next button', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Next/)).toBeDefined()
  })

  it('shows hint when no dataset selected', () => {
    render(<DataStep form={form} datasets={datasets} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/Select a dataset/)).toBeDefined()
  })

  it('shows preview when dataset has samples', () => {
    const datasetsWithPreview: UseTrainingDatasetsReturn = {
      ...datasets,
      datasetPreview: {
        dataset_id: 'ds-1',
        samples: [
          { path: 'a.txt', language: 'text', content: 'line 1', size: 6 },
          { path: 'b.txt', language: 'text', content: 'line 2', size: 6 },
        ],
        total_samples: 100,
        total_chars: 5000,
        languages: { text: 100 },
      },
    }
    render(
      <DataStep form={form} datasets={datasetsWithPreview} onNext={vi.fn()} onBack={vi.fn()} />,
    )
    expect(screen.getByText('Preview')).toBeDefined()
    expect(screen.getByText('line 1')).toBeDefined()
  })

  it('hides hint when dataset is selected', () => {
    const ds: UseTrainingDatasetsReturn = { ...datasets, selectedDataset: 'shakespeare' }
    render(<DataStep form={form} datasets={ds} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.queryByText(/Select a dataset/)).toBeNull()
  })

  it('passes selectedDataset to DatasetSelector', () => {
    const ds: UseTrainingDatasetsReturn = { ...datasets, selectedDataset: 'my-dataset' }
    render(<DataStep form={form} datasets={ds} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toHaveTextContent('my-dataset')
  })

  it('hides system-kind entries from the chip cloud', () => {
    const mixed = makeDatasets({
      datasets: [
        ds({ id: 'api_conversations', name: 'Api Conversations', kind: 'dataset', samples: 1662 }),
        ds({ id: 'auth_mogdb', name: 'Auth Mogdb', kind: 'system' }),
        ds({ id: 'backups', name: 'Backups', kind: 'system' }),
        ds({ id: 'knowledge_json', name: 'Knowledge Json', kind: 'system' }),
      ],
    })
    render(<DataStep form={form} datasets={mixed} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Api Conversations')).toBeDefined()
    expect(screen.queryByText('Auth Mogdb')).toBeNull()
    expect(screen.queryByText('Backups')).toBeNull()
    expect(screen.queryByText('Knowledge Json')).toBeNull()
  })

  it('keeps untagged legacy entries in the chip cloud', () => {
    const legacy = makeDatasets({
      datasets: [ds({ id: 'old-ds', name: 'old-ds' })],
    })
    render(<DataStep form={form} datasets={legacy} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Old Ds')).toBeDefined()
  })

  it('humanizes chip labels and exposes raw id via aria-label', () => {
    const named = makeDatasets({
      datasets: [
        ds({ id: 'mental-health-chat-200k', name: 'mental_health_chat_200k', kind: 'dataset' }),
      ],
    })
    render(<DataStep form={form} datasets={named} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText('Mental Health Chat 200k')).toBeDefined()
    expect(screen.getByLabelText('mental-health-chat-200k')).toBeDefined()
  })

  it('shows system-stores message when only system entries exist', () => {
    const onlySystem = makeDatasets({
      datasets: [
        ds({ id: 'auth_mogdb', name: 'Auth Mogdb', kind: 'system' }),
        ds({ id: 'backups', name: 'Backups', kind: 'system' }),
      ],
    })
    render(<DataStep form={form} datasets={onlySystem} onNext={vi.fn()} onBack={vi.fn()} />)
    expect(screen.getByText(/System stores hidden/)).toBeDefined()
    expect(screen.getByText(/Only system stores found/)).toBeDefined()
    expect(screen.queryByText('Auth Mogdb')).toBeNull()
  })
})
