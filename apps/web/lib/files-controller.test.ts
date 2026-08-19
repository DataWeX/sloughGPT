import { describe, it, expect, vi, beforeEach } from 'vitest'
import { filesController } from './files-controller'
import * as http from './http-client'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiDelete: vi.fn(),
}))

const apiGet = vi.mocked(http.apiGet)
const apiPost = vi.mocked(http.apiPost)
const apiDelete = vi.mocked(http.apiDelete)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('filesController', () => {
  const mockFiles = [
    { id: '1', filename: 'test.txt', size: 100, content_type: 'text/plain', uploaded_at: '2026-01-01', ingested: false },
    { id: '2', filename: 'data.csv', size: 200, content_type: 'text/csv', uploaded_at: '2026-01-02', ingested: true, chunk_count: 5 },
  ]

  it('list returns files from {files} response', async () => {
    apiGet.mockResolvedValue({ files: mockFiles })
    const result = await filesController.list()
    expect(result).toEqual(mockFiles)
    expect(apiGet).toHaveBeenCalledWith('/files/')
  })

  it('list handles flat array response', async () => {
    apiGet.mockResolvedValue(mockFiles)
    const result = await filesController.list()
    expect(result).toEqual(mockFiles)
  })

  it('list handles empty response', async () => {
    apiGet.mockResolvedValue(undefined)
    const result = await filesController.list()
    expect(result).toEqual([])
  })

  it('upload calls apiPost with raw option', async () => {
    const formData = new FormData()
    apiPost.mockResolvedValue({ filename: 'test.txt' })
    const result = await filesController.upload(formData)
    expect(result).toEqual({ filename: 'test.txt' })
    expect(apiPost).toHaveBeenCalledWith('/files/upload', formData, { raw: true })
  })

  it('delete calls apiDelete with correct path', async () => {
    apiDelete.mockResolvedValue(undefined)
    await filesController.delete('abc-123')
    expect(apiDelete).toHaveBeenCalledWith('/files/abc-123')
  })

  it('ingest calls apiPost with correct path', async () => {
    apiPost.mockResolvedValue(undefined)
    await filesController.ingest('abc-123')
    expect(apiPost).toHaveBeenCalledWith('/files/abc-123/ingest')
  })

  it('search calls apiGet with encoded query', async () => {
    apiGet.mockResolvedValue({ results: [mockFiles[0]] })
    const result = await filesController.search('hello world')
    expect(result).toEqual([mockFiles[0]])
    expect(apiGet).toHaveBeenCalledWith('/files/search?q=hello%20world')
  })

  it('search handles flat array response', async () => {
    apiGet.mockResolvedValue(mockFiles)
    const result = await filesController.search('test')
    expect(result).toEqual(mockFiles)
  })

  it('search handles empty/undefined response', async () => {
    apiGet.mockResolvedValue(undefined)
    const result = await filesController.search('test')
    expect(result).toEqual([])
  })
})
