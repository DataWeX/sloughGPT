import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockApiPost = vi.fn()
const mockApiGet = vi.fn()
const mockFetch = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiGet: (...args: unknown[]) => mockApiGet(...args),
}))

vi.stubGlobal('fetch', mockFetch)

const { worldController } = await import('@/lib/world-controller')

describe('worldController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('render calls POST /world/render with config', async () => {
    mockApiPost.mockResolvedValue({ shapes: { world: [1, 2] }, stats: {}, tensor_keys: ['world'] })
    const result = await worldController.render({ width: 80, height: 60 })
    expect(mockApiPost).toHaveBeenCalledWith('/world/render', { width: 80, height: 60 })
    expect(result.tensor_keys).toEqual(['world'])
  })

  it('render uses empty config when none provided', async () => {
    mockApiPost.mockResolvedValue({ shapes: {}, stats: {}, tensor_keys: [] })
    await worldController.render()
    expect(mockApiPost).toHaveBeenCalledWith('/world/render', {})
  })

  it('renderImage calls fetch and returns blob', async () => {
    const fakeBlob = new Blob(['image data'], { type: 'image/png' })
    mockFetch.mockResolvedValue({ ok: true, blob: () => Promise.resolve(fakeBlob) })
    const result = await worldController.renderImage({ width: 80 })
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/world/render/image'),
      expect.objectContaining({ method: 'POST' })
    )
    expect(result).toBe(fakeBlob)
  })

  it('renderImage throws on non-ok response', async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 500 })
    await expect(worldController.renderImage()).rejects.toThrow('Render failed')
  })

  it('tick calls POST /world/tick with params', async () => {
    mockApiPost.mockResolvedValue({ tick: 1, babies: 3, render_stats: null })
    const result = await worldController.tick(2, true, true)
    expect(mockApiPost).toHaveBeenCalledWith('/world/tick', { max_ticks: 2, render: true, neural: true })
    expect(result.tick).toBe(1)
    expect(result.babies).toBe(3)
  })

  it('tick uses defaults', async () => {
    mockApiPost.mockResolvedValue({ tick: 1, babies: 0, render_stats: null })
    await worldController.tick()
    expect(mockApiPost).toHaveBeenCalledWith('/world/tick', { max_ticks: 1, render: true, neural: false })
  })

  it('neuralProcess calls POST /world/neural', async () => {
    mockApiPost.mockResolvedValue({ embedding_shape: [64], descriptor: { key: 'val' }, stats: {} })
    const result = await worldController.neuralProcess({ width: 100 })
    expect(mockApiPost).toHaveBeenCalledWith('/world/neural', { width: 100 })
    expect(result.embedding_shape).toEqual([64])
  })

  it('stats calls GET /world/stats', async () => {
    mockApiGet.mockResolvedValue({ status: 'ok', components: ['physics'], materials: { grass: 0 } })
    const result = await worldController.stats()
    expect(mockApiGet).toHaveBeenCalledWith('/world/stats')
    expect(result.components).toEqual(['physics'])
  })

  it('propagates errors from apiPost', async () => {
    mockApiPost.mockRejectedValue(new Error('OOM'))
    await expect(worldController.render()).rejects.toThrow('OOM')
  })

  it('propagates errors from apiGet', async () => {
    mockApiGet.mockRejectedValue(new Error('timeout'))
    await expect(worldController.stats()).rejects.toThrow('timeout')
  })
})
