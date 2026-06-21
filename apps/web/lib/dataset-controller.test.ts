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

import { datasetController } from './dataset-controller'

describe('datasetController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets and returns rows', async () => {
    apiClient.apiGet.mockResolvedValue({ datasets: [{ id: 'ds1', name: 'shakespeare', source: 'local', size: 12345, samples: 100, type: 'text', created_at: '2026-01-01' }] })
    const rows = await datasetController.list()
    expect(rows).toHaveLength(1)
    expect(rows[0].id).toBe('ds1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets')
  })

  it('handles empty datasets', async () => {
    apiClient.apiGet.mockResolvedValue({ datasets: [] })
    const rows = await datasetController.list()
    expect(rows).toEqual([])
  })

  it('returns empty when datasets field missing', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const rows = await datasetController.list()
    expect(rows).toEqual([])
  })

  it('throws on API error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('502'))
    await expect(datasetController.list()).rejects.toThrow('502')
  })
})

describe('datasetController.get', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets/{id}', async () => {
    apiClient.apiGet.mockResolvedValue({ id: 'ds1', name: 'shakespeare', source: 'local', size: 12345, created_at: '2026-01-01' })
    const result = await datasetController.get('ds1')
    expect(result).not.toBeNull()
    expect(result!.name).toBe('shakespeare')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds1')
  })

  it('returns null on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('404'))
    const result = await datasetController.get('nonexistent')
    expect(result).toBeNull()
  })
})

describe('datasetController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /datasets', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 'ds2', name: 'my-data', source: 'manual', size: 0, created_at: '2026-01-01' })
    const result = await datasetController.create({ name: 'my-data' })
    expect(result.name).toBe('my-data')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets', { name: 'my-data' })
  })
})

describe('datasetController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /datasets/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue({})
    await datasetController.delete('ds1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/datasets/ds1')
  })
})

describe('datasetController.addData', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /datasets/{id}/data with strings', async () => {
    apiClient.apiPost.mockResolvedValue({})
    await datasetController.addData('ds1', ['line1', 'line2'])
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/ds1/data', { data: ['line1', 'line2'] })
  })
})

describe('datasetController.update', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PATCHes /datasets/{id}', async () => {
    apiClient.apiPatch.mockResolvedValue({ id: 'ds1', name: 'renamed', source: 'local', size: 100, created_at: '2026-01-01' })
    const result = await datasetController.update('ds1', { name: 'renamed' })
    expect(result.name).toBe('renamed')
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/datasets/ds1', { name: 'renamed' })
  })
})

describe('datasetController.preview', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets/{id}/preview', async () => {
    const preview = { dataset_id: 'ds1', samples: [{ path: 'file.txt', language: 'en', content: 'hello', size: 5 }], total_samples: 1, total_chars: 5, languages: { en: 1 } }
    apiClient.apiGet.mockResolvedValue(preview)
    const result = await datasetController.preview('ds1', 5)
    expect(result.total_samples).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds1/preview?limit=5')
  })
})

describe('datasetController.searchGitHubRepos', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets/search/github', async () => {
    apiClient.apiGet.mockResolvedValue({ repos: [{ id: 'r1', name: 'repo', full_name: 'user/repo', description: 'desc', stars: 10, url: 'https://github.com/user/repo', language: 'Python' }] })
    const result = await datasetController.searchGitHubRepos('python')
    expect(result.repos).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/search/github', { q: 'python', limit: '10' })
  })
})

describe('datasetController.searchBooks', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets/search/books', async () => {
    apiClient.apiGet.mockResolvedValue({ books: [{ key: 'b1', title: 'Moby Dick', author: 'Melville', isbn: '123', year: 1851, cover: null }] })
    const result = await datasetController.searchBooks('moby', 5)
    expect(result.books).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/search/books', { q: 'moby', limit: '5' })
  })
})

describe('datasetController.imports', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('importFromGitHub POSTs /datasets/import/github', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds1', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromGitHub({ url: 'https://github.com/user/repo', name: 'repo' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/github', { url: 'https://github.com/user/repo', name: 'repo' })
  })

  it('importFromHuggingFace POSTs /datasets/import/huggingface', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds2', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromHuggingFace({ dataset_id: 'imdb', name: 'my-imdb' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/huggingface', { dataset_id: 'imdb', name: 'my-imdb' })
  })

  it('importFromURL POSTs /datasets/import/url', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds3', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromURL({ url: 'https://data.gov/sample.jsonl', name: 'gov-data' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/url', { url: 'https://data.gov/sample.jsonl', name: 'gov-data' })
  })

  it('importFromLocal POSTs /datasets/import/local', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds4', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromLocal({ path: '/data/txt', name: 'local-data' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/local', { path: '/data/txt', name: 'local-data' })
  })

  it('importFromKaggle POSTs /datasets/import/kaggle', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds5', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromKaggle({ dataset: 'user/dataset', name: 'kaggle-data' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/kaggle', { dataset: 'user/dataset', name: 'kaggle-data' })
  })

  it('importFromCSV POSTs /datasets/import/csv', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds6', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromCSV({ url: 'https://example.com/data.csv', name: 'csv-data', delimiter: ';' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/csv', { url: 'https://example.com/data.csv', name: 'csv-data', delimiter: ';' })
  })

  it('importFromISBN POSTs /datasets/import/isbn', async () => {
    apiClient.apiPost.mockResolvedValue({ success: true, dataset_id: 'ds7', message: 'ok', output_path: '/tmp' })
    const r = await datasetController.importFromISBN({ isbn: '9780141036144', name: '1984-book' })
    expect(r.success).toBe(true)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/isbn', { isbn: '9780141036144', name: '1984-book' })
  })

  it('batchImport POSTs /datasets/import/batch', async () => {
    apiClient.apiPost.mockResolvedValue({ imported: 2, errors: [] })
    const r = await datasetController.batchImport(['src1', 'src2'])
    expect(r.imported).toBe(2)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/import/batch', { sources: ['src1', 'src2'] })
  })
})

describe('datasetController.export', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('fetches blob from /datasets/{id}/export', async () => {
    const fakeBlob = new Blob(['data,a,b\n1,2,3'])
    global.fetch = vi.fn().mockResolvedValue({ blob: () => fakeBlob })
    const result = await datasetController.export('ds1', 'csv')
    expect(result).toBe(fakeBlob)
    expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:9/datasets/ds1/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ format: 'csv' }),
    })
  })
})

describe('datasetController.convertToMessages', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /datasets/convert-to-messages', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', new_dataset_id: 'ds2', total_conversations: 5 })
    const r = await datasetController.convertToMessages('ds1', 'You are helpful.')
    expect(r.total_conversations).toBe(5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/convert-to-messages?dataset_id=ds1&system_prompt=You%20are%20helpful.')
  })
})

describe('datasetController.getStats', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /datasets/{id}/stats', async () => {
    apiClient.apiGet.mockResolvedValue({ format: 'jsonl', samples: 100, chars: 5000, avg_length: 50, has_messages: false, sample_preview: [], lines: 100, suggested_method: 'distill', file_type: 'jsonl' })
    const r = await datasetController.getStats('ds1')
    expect(r.samples).toBe(100)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds1/stats')
  })

  it('encodes special characters in id', async () => {
    apiClient.apiGet.mockResolvedValue({ format: 'text', samples: 0, chars: 0, avg_length: 0, has_messages: false, sample_preview: [], lines: 0, suggested_method: '', file_type: '' })
    await datasetController.getStats('ds/path')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/datasets/ds%2Fpath/stats')
  })
})

describe('datasetController.createFromChat', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /datasets/from-chat', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', dataset_id: 'ds1', name: 'chat-export', messages_exported: 3 })
    const r = await datasetController.createFromChat({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }], name: 'chat-export' })
    expect(r.messages_exported).toBe(3)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/datasets/from-chat', {
      messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }], name: 'chat-export',
    })
  })
})
