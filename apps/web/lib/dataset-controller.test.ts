import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import {
  datasetController,
  humanizeDatasetName,
  isTrainingCorpus,
  sanitizeDatasetName,
} from './dataset-controller'

describe('isTrainingCorpus', () => {
  it('keeps dataset and untagged entries', () => {
    expect(isTrainingCorpus({ kind: 'dataset' })).toBe(true)
    expect(isTrainingCorpus({ kind: undefined })).toBe(true)
    expect(isTrainingCorpus({ kind: null as unknown as string })).toBe(true)
  })

  it('rejects adapter/system/media kinds', () => {
    expect(isTrainingCorpus({ kind: 'adapter' })).toBe(false)
    expect(isTrainingCorpus({ kind: 'system' })).toBe(false)
    expect(isTrainingCorpus({ kind: 'media' })).toBe(false)
  })
})

describe('humanizeDatasetName', () => {
  it('replaces underscores and hyphens with spaces', () => {
    expect(humanizeDatasetName('mental_health-chat')).toBe('Mental Health Chat')
  })

  it('strips storage-backend suffixes', () => {
    expect(humanizeDatasetName('knowledge_graph_json')).toBe('Knowledge Graph')
    expect(humanizeDatasetName('Auth-Mogdb')).toBe('Auth')
    expect(humanizeDatasetName('errors_json')).toBe('Errors')
    expect(humanizeDatasetName('model_catalog_json')).toBe('Model Catalog')
  })

  it('title-cases lowercase words only', () => {
    expect(humanizeDatasetName('tinyshakespeare')).toBe('Tinyshakespeare')
    expect(humanizeDatasetName('Api Conversations')).toBe('Api Conversations')
    expect(humanizeDatasetName('ultrachat_200k')).toBe('Ultrachat 200k')
  })

  it('returns original when empty after cleanup', () => {
    expect(humanizeDatasetName('_')).toBe('_')
    expect(humanizeDatasetName('')).toBe('')
  })
})

describe('sanitizeDatasetName', () => {
  it('strips the file extension', () => {
    expect(sanitizeDatasetName('train-src.txt')).toBe('train-src')
    expect(sanitizeDatasetName('notes.v2.md')).toBe('notes_v2')
  })

  it('replaces characters the dataset-id validator rejects', () => {
    expect(sanitizeDatasetName('my notes (final).csv')).toBe('my_notes_final')
    expect(sanitizeDatasetName('data 2026.csv')).toBe('data_2026')
  })

  it('never returns empty', () => {
    expect(sanitizeDatasetName('.txt')).toBe('file')
    expect(sanitizeDatasetName('🎉.png')).toBe('file')
    expect(sanitizeDatasetName('')).toBe('file')
  })
})

describe('datasetController.list', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('GET /datasets and returns rows', async () => {
    const mockData = {
      datasets: [
        {
          id: 'ds1',
          name: 'shakespeare',
          source: 'local',
          size: 12345,
          samples: 100,
          type: 'text',
          created_at: '2026-01-01',
        },
      ],
    }
    apiClient.apiGet.mockResolvedValue(mockData)

    const rows = await datasetController.list()

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets')
    expect(rows).toHaveLength(1)
    expect(rows[0].id).toBe('ds1')
    expect(rows[0].name).toBe('shakespeare')
  })

  it('handles empty datasets', async () => {
    apiClient.apiGet.mockResolvedValue({ datasets: [] })

    const rows = await datasetController.list()
    expect(rows).toEqual([])
  })

  it('throws when GET /datasets is not ok', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('502'))

    await expect(datasetController.list()).rejects.toThrow('502')
  })
})

describe('datasetController.search', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('GETs /datasets/search with the query and returns results', async () => {
    const results = [
      { id: 'ds2', name: 'shakespeare', source: 'local', size: 10, created_at: '2026-01-02' },
    ]
    apiClient.apiGet.mockResolvedValue({ results, count: 1 })

    const rows = await datasetController.search('shakes')

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/search?q=shakes')
    expect(rows).toEqual(results)
  })

  it('encodes special characters in the query', async () => {
    apiClient.apiGet.mockResolvedValue({ results: [], count: 0 })

    await datasetController.search('s&p 500 dataset')

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/search?q=s%26p%20500%20dataset')
  })

  it('returns empty array when no results key', async () => {
    apiClient.apiGet.mockResolvedValue({})

    const rows = await datasetController.search('nothing')

    expect(rows).toEqual([])
  })
})

describe('datasetController.export', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('POSTs to /datasets/{id}/export and returns the blob', async () => {
    const blob = new Blob(['{"a":1}'], { type: 'application/json' })
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => blob })
    vi.stubGlobal('fetch', fetchMock)

    const result = await datasetController.export('ds1', 'jsonl')

    expect(fetchMock).toHaveBeenCalledWith(
      '/datasets/ds1/export',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ format: 'jsonl' }),
      }),
    )
    expect(result).toBe(blob)
    vi.unstubAllGlobals()
  })

  it('defaults to jsonl format', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, blob: async () => new Blob() })
    vi.stubGlobal('fetch', fetchMock)

    await datasetController.export('ds1')

    expect(fetchMock).toHaveBeenCalledWith(
      '/datasets/ds1/export',
      expect.objectContaining({ body: JSON.stringify({ format: 'jsonl' }) }),
    )
    vi.unstubAllGlobals()
  })
})

describe('datasetController versioning', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('creates a version via POST /datasets/{id}/versions', async () => {
    apiClient.apiPost.mockResolvedValue({ timestamp: '20260801120000', message: 'Version created' })

    const res = await datasetController.createVersion('ds1')

    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/ds1/versions')
    expect(res.timestamp).toBe('20260801120000')
  })

  it('lists versions via GET /datasets/{id}/versions', async () => {
    apiClient.apiGet.mockResolvedValue({ versions: ['20260801120000', '20260801110000'], count: 2 })

    const res = await datasetController.listVersions('ds1')

    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds1/versions')
    expect(res.versions).toHaveLength(2)
    expect(res.count).toBe(2)
  })

  it('restores a version via POST /datasets/{id}/versions/{timestamp}', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, message: 'Version restored' })

    const res = await datasetController.restoreVersion('ds1', '20260801120000')

    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/ds1/versions/20260801120000')
    expect(res.success).toBe(true)
  })
})
