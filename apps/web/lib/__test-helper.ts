/**
 * Shared test helper — mocks `./http-client` so `api.ts` methods use fake axios.
 * Tests assert against `apiGet`/`apiPost` spy args instead of raw `fetch` calls.
 *
 * Usage:
 *   import { setupApiMocks, mockGet, mockPost, apiClient } from '../__test-helper'
 *   setupApiMocks()
 *   mockGet('/health', { status: 'ok' })
 *   const result = await api.getHealth()
 *   expect(apiClient.apiGet).toHaveBeenCalledWith('/health')
 */

import { vi } from 'vitest'

// These are created before the mock factory runs via vi.hoisted
const { mockApiGet, mockApiPost, mockApiPut, mockApiDelete, mockApiPatch }
  = vi.hoisted(() => ({
    mockApiGet: vi.fn(),
    mockApiPost: vi.fn(),
    mockApiPut: vi.fn(),
    mockApiDelete: vi.fn(),
    mockApiPatch: vi.fn(),
  }))

// Hoisted mock – runs at top level, before any imports
vi.mock('./http-client', () => {
  const makeClient = () => ({
    defaults: { baseURL: 'http://127.0.0.1:9' },
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
    interceptors: { request: { use: vi.fn(), eject: vi.fn() }, response: { use: vi.fn(), eject: vi.fn() } },
    headers: { common: {} as Record<string, string> },
  })

  return {
    __esModule: true,
    apiGet: mockApiGet,
    apiPost: mockApiPost,
    apiPut: mockApiPut,
    apiDelete: mockApiDelete,
    apiPatch: mockApiPatch,
    apiClient: makeClient(),
    createApiClient: vi.fn(() => makeClient()),
  }
})

export function setupApiMocks() {
  // No-op: mock is already in place at top level
}

export function mockGet<T>(url: string, data: T): void {
  mockApiGet.mockImplementation((u: string) => {
    if (u === url) return data
    throw new Error(`Unexpected apiGet call: ${u}`)
  })
}

export function mockPost<T>(url: string, data: T): void {
  mockApiPost.mockImplementation((u: string) => {
    if (u === url) return data
    throw new Error(`Unexpected apiPost call: ${u}`)
  })
}

export function mockDelete<T>(url: string, data: T): void {
  mockApiDelete.mockImplementation((u: string) => {
    if (u === url) return data
    throw new Error(`Unexpected apiDelete call: ${u}`)
  })
}

export const apiClient = {
  apiGet: mockApiGet,
  apiPost: mockApiPost,
  apiPut: mockApiPut,
  apiDelete: mockApiDelete,
  apiPatch: mockApiPatch,
}
