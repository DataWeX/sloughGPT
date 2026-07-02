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

import { filesController } from './files-controller'

describe('filesController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list GETs /files', async () => {
    apiClient.apiGet.mockResolvedValue({ files: [{ id: '1', filename: 'test.txt' }], total: 1 })
    const result = await filesController.list()
    expect(result.files.length).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files')
  })

  it('list passes sort/order/tag params', async () => {
    apiClient.apiGet.mockResolvedValue({ files: [], total: 0 })
    await filesController.list('name', 'asc', 'code')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files?sort=name&order=asc&tag=code')
  })

  it('upload POSTs multipart form data', async () => {
    apiClient.apiPost.mockResolvedValue({ id: '1', filename: 'test.txt', chars: 100, pages: 1, size_bytes: 200 })
    const file = new File(['hello'], 'test.txt', { type: 'text/plain' })
    const result = await filesController.upload(file)
    expect(result.id).toBe('1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/files/upload', expect.any(FormData), { raw: true })
  })

  it('upload passes tags in form data', async () => {
    apiClient.apiPost.mockResolvedValue({ id: '2', filename: 'doc.txt', chars: 50, pages: 1, size_bytes: 100 })
    const file = new File(['data'], 'doc.txt')
    await filesController.upload(file, ['code', 'docs'])
    const callArgs = apiClient.apiPost.mock.calls[0][1] as FormData
    expect(callArgs.get('tags')).toBe('["code","docs"]')
  })

  it('get GETs /files/{id}', async () => {
    apiClient.apiGet.mockResolvedValue({ id: '1', filename: 'test.txt', extension: '.txt', size_bytes: 200, chars: 100, pages: 1, uploaded_at: 1000, tags: [], text: 'hello' })
    const result = await filesController.get('1')
    expect(result.filename).toBe('test.txt')
    expect(result.text).toBe('hello')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files/1')
  })

  it('delete DELETEs /files/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)
    await filesController.delete('1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/files/1')
  })

  it('search GETs /files/search', async () => {
    apiClient.apiGet.mockResolvedValue({ files: [{ id: '1', filename: 'test.txt' }], total: 1 })
    const result = await filesController.search('hello')
    expect(result.files.length).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files/search?q=hello')
  })

  it('search passes tag param', async () => {
    apiClient.apiGet.mockResolvedValue({ files: [], total: 0 })
    await filesController.search('hello', 'code')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files/search?q=hello&tag=code')
  })

  it('ingest POSTs /files/{id}/ingest', async () => {
    apiClient.apiPost.mockResolvedValue({ id: '1', filename: 'test.txt', chars: 100, facts_stored: 5 })
    const result = await filesController.ingest('1')
    expect(result.facts_stored).toBe(5)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/files/1/ingest')
  })

  it('extract uploads then gets file detail', async () => {
    apiClient.apiPost.mockResolvedValue({ id: '1', filename: 'test.txt', chars: 100, pages: 1, size_bytes: 200 })
    apiClient.apiGet.mockResolvedValue({ id: '1', filename: 'test.txt', extension: '.txt', size_bytes: 200, chars: 100, pages: 1, uploaded_at: 1000, tags: [], text: 'extracted content' })
    const file = new File(['hello'], 'test.txt')
    const result = await filesController.extract(file)
    expect(result.text).toBe('extracted content')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/files/upload', expect.any(FormData), { raw: true })
    expect(apiClient.apiGet).toHaveBeenCalledWith('/files/1')
  })

  describe('formatSize', () => {
    it('returns B for small sizes', () => {
      expect(filesController.formatSize(500)).toBe('500 B')
    })
    it('returns KB for medium sizes', () => {
      expect(filesController.formatSize(2048)).toBe('2.0 KB')
    })
    it('returns MB for large sizes', () => {
      expect(filesController.formatSize(5 * 1024 * 1024)).toBe('5.0 MB')
    })
  })

  describe('formatDate', () => {
    it('formats unix timestamp', () => {
      const result = filesController.formatDate(0)
      expect(result).toContain('1970')
    })
  })
})
