import { describe, it, expect, beforeEach, vi } from 'vitest'
import { apiGet, apiPost, apiPut, apiPatch, apiDelete, ApiError } from '@/lib/api-client'

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(global.fetch).mockClear()
  })

  it('should make GET request', async () => {
    const mockData = { id: '1', name: 'Test' }
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve(mockData),
    } as any)

    const result = await apiGet('/test')

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/test'),
      expect.objectContaining({ method: 'GET' })
    )
    expect(result).toEqual(mockData)
  })

  it('should make POST request with body', async () => {
    const requestBody = { name: 'Test' }
    const responseData = { id: '1', ...requestBody }
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve(responseData),
    } as any)

    const result = await apiPost('/test', requestBody)

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/test'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(requestBody),
      })
    )
    expect(result).toEqual(responseData)
  })

  it('should make PUT request', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ status: 'updated' }),
    } as any)

    await apiPut('/test/1', { name: 'Updated' })

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/test/1'),
      expect.objectContaining({ method: 'PUT' })
    )
  })

  it('should make PATCH request', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ status: 'patched' }),
    } as any)

    await apiPatch('/test/1', { field: 'value' })

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/test/1'),
      expect.objectContaining({ method: 'PATCH' })
    )
  })

  it('should make DELETE request', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ status: 'deleted' }),
    } as any)

    await apiDelete('/test/1')

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/test/1'),
      expect.objectContaining({ method: 'DELETE' })
    )
  })

  it('should throw ApiError on non-OK response', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: false,
      status: 404,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ error: 'Not found' }),
    } as any)

    await expect(apiGet('/not-found')).rejects.toThrow(ApiError)
  })

  it('should retry on 503 error', async () => {
    let callCount = 0
    vi.mocked(global.fetch).mockImplementation(() => {
      callCount++
      if (callCount < 3) {
        return Promise.resolve({
          ok: false,
          status: 503,
          headers: { get: () => 'application/json' },
          json: () => Promise.resolve({ error: 'Service unavailable' }),
        } as any)
      }
      return Promise.resolve({
        ok: true,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ status: 'ok' }),
      } as any)
    })

    const result = await apiGet('/test')

    expect(callCount).toBe(3)
    expect(result).toEqual({ status: 'ok' })
  })

  it('should handle text response', async () => {
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      headers: { get: () => 'text/plain' },
      text: () => Promise.resolve('Plain text response'),
    } as any)

    const result = await apiGet('/text')

    expect(result).toBe('Plain text response')
  })
})
