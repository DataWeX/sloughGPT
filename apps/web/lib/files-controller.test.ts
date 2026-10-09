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
  const backendFiles = [
    { id: '1', filename: 'test.txt', extension: 'txt', size_bytes: 100, uploaded_at: '2026-01-01', tags: ['doc'] },
    { id: '2', filename: 'data.csv', extension: 'csv', size_bytes: 200, uploaded_at: '2026-01-02', tags: [] },
  ]

  it('list returns mapped files from array response', async () => {
    apiGet.mockResolvedValue(backendFiles)
    const result = await filesController.list()
    expect(result[0].id).toBe('1')
    expect(result[0].size).toBe(100)
    expect(result[0].content_type).toBe('txt')
    expect(apiGet).toHaveBeenCalledWith('/files/')
  })

  it('list handles {files} response', async () => {
    apiGet.mockResolvedValue({ files: backendFiles })
    const result = await filesController.list()
    expect(result).toHaveLength(2)
    expect(result[0].id).toBe('1')
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
    apiGet.mockResolvedValue({ files: [backendFiles[0]] })
    const result = await filesController.search('hello world')
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('1')
    expect(apiGet).toHaveBeenCalledWith('/files/search?q=hello%20world')
  })

  it('search handles {files} response', async () => {
    apiGet.mockResolvedValue({ files: backendFiles })
    const result = await filesController.search('test')
    expect(result).toHaveLength(2)
  })

  it('search handles empty/undefined response', async () => {
    apiGet.mockResolvedValue(undefined)
    const result = await filesController.search('test')
    expect(result).toEqual([])
  })
})
