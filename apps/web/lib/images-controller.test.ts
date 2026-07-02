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

import { imagesController } from './images-controller'

describe('imagesController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('generate POSTs /images/generate with prompt and style', async () => {
    apiClient.apiPost.mockResolvedValue({ image: 'data:base64,...', style: 'realistic', prompt: 'a cat', id: '1' })
    const result = await imagesController.generate('a cat', 'realistic')
    expect(result.id).toBe('1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/images/generate', { prompt: 'a cat', style: 'realistic' })
  })

  it('generate uses default style', async () => {
    apiClient.apiPost.mockResolvedValue({ image: 'data:base64,...', style: 'realistic', prompt: 'dog', id: '2' })
    await imagesController.generate('dog')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/images/generate', { prompt: 'dog', style: 'realistic' })
  })

  it('gallery GETs /images/gallery', async () => {
    apiClient.apiGet.mockResolvedValue({ images: [{ id: '1', path: '/img/cat.png', created: 1000 }] })
    const result = await imagesController.gallery()
    expect(result.images.length).toBe(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/images/gallery')
  })

  it('styles GETs /images/styles', async () => {
    apiClient.apiGet.mockResolvedValue({ styles: [['realistic', 'Realistic'], ['cartoon', 'Cartoon']] })
    const result = await imagesController.styles()
    expect(result.styles.length).toBe(2)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/images/styles')
  })

  it('handles generate error', async () => {
    apiClient.apiPost.mockRejectedValue(new Error('Generation failed'))
    await expect(imagesController.generate('fail')).rejects.toThrow('Generation failed')
  })
})
