import React from 'react'
import { render, fireEvent, waitFor } from '../../test-utils'
import { ImportDatasetModal } from '../training/ImportDatasetModal'
import { searchDatasets } from '../../services/training-service'

const mockImportDataset = jest.fn(async () => {})

jest.mock('../../services/training-service', () => ({
  searchDatasets: jest.fn(),
}))

jest.mock('../../stores/training-store', () => ({
  useTrainingStore: (selector: any) => selector({ importDataset: mockImportDataset }),
}))

jest.mock('../../services/toast', () => ({
  toast: { error: jest.fn(), success: jest.fn(), info: jest.fn(), warn: jest.fn() },
}))

jest.mock('../../services/haptics', () => ({
  triggerHaptic: jest.fn(),
}))

jest.mock('../Icon', () => ({
  Icon: ({ name }: any) => <div data-testid={`icon-${name}`} />,
}))

beforeEach(() => {
  jest.clearAllMocks()
})

const GITHUB_RESULT = {
  source: 'github' as const,
  id: 'owner/vision-datasets',
  title: 'owner/vision-datasets',
  description: 'image-caption pairs',
  stats: { stars: 120, language: 'Python', url: '' },
}

const KAGGLE_RESULT = {
  source: 'kaggle' as const,
  id: 'competitions/titanic',
  title: 'Titanic',
  description: null,
  stats: { downloads: 1, votes: 2 },
}

function searchResponse(overrides?: Partial<{ results: any[]; providers: any[] }>) {
  return {
    results: [GITHUB_RESULT, KAGGLE_RESULT],
    providers: [
      { source: 'github', ok: true, count: 1, error: null, elapsed_ms: 120 },
      { source: 'huggingface', ok: true, count: 0, error: null, elapsed_ms: 90 },
      {
        source: 'kaggle',
        ok: false,
        count: 0,
        error: 'kaggle CLI not found',
        elapsed_ms: 5,
      },
    ],
    ...overrides,
  }
}

async function runSearch(query: string) {
  fireEvent.changeText(renderResult.getByTestId('federated-search-input'), query)
  await waitFor(() =>
    expect(renderResult.getByTestId('federated-search-input').props.value).toBe(query),
  )
  fireEvent.press(renderResult.getByText('Search'))
  await waitFor(() => expect(searchDatasets).toHaveBeenCalled())
  await waitFor(() => expect(renderResult.getByText('owner/vision-datasets')).toBeTruthy())
}

let renderResult: ReturnType<typeof render>

describe('ImportDatasetModal — federated search parity', () => {
  it('renders the search-all-sources panel above the manual sources', async () => {
    renderResult = await render(<ImportDatasetModal visible onClose={jest.fn()} />)
    expect(renderResult.getByText('Import Dataset')).toBeTruthy()
    expect(renderResult.getByText('Search all sources')).toBeTruthy()
    expect(renderResult.getByText('URL')).toBeTruthy()
    expect(renderResult.getByText('GitHub')).toBeTruthy()
  })

  it('searches one query across sources and renders partial provider status', async () => {
    ;(searchDatasets as jest.Mock).mockResolvedValue(searchResponse())
    renderResult = await render(<ImportDatasetModal visible onClose={jest.fn()} />)

    await runSearch('vision models')

    expect(searchDatasets).toHaveBeenCalledWith('vision models')
    expect(renderResult.getByText('kaggle: kaggle CLI not found')).toBeTruthy()
    expect(renderResult.getByText('not importable on mobile')).toBeTruthy()
  })

  it('selecting a github result fills source + name and imports', async () => {
    ;(searchDatasets as jest.Mock).mockResolvedValue(searchResponse())
    renderResult = await render(<ImportDatasetModal visible onClose={jest.fn()} />)
    await runSearch('vision models')

    fireEvent.press(renderResult.getByLabelText('Select owner/vision-datasets from github'))
    await waitFor(() =>
      expect(renderResult.getByTestId('import-source-input').props.value).toBe(
        'owner/vision-datasets',
      ),
    )

    fireEvent.press(renderResult.getByText('Import'))
    await waitFor(() =>
      expect(mockImportDataset).toHaveBeenCalledWith(
        'owner/vision-datasets',
        'vision-datasets',
        'github',
      ),
    )
  })

  it('kaggle/openlibrary rows are non-selectable on mobile', async () => {
    ;(searchDatasets as jest.Mock).mockResolvedValue(searchResponse())
    renderResult = await render(<ImportDatasetModal visible onClose={jest.fn()} />)
    await runSearch('vision models')

    fireEvent.press(renderResult.getByLabelText('Select Titanic from kaggle'))

    fireEvent.press(renderResult.getByText('Import'))
    expect(mockImportDataset).not.toHaveBeenCalled()
  })
})
