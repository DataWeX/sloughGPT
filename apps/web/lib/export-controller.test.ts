import { describe, it, expect, vi, beforeEach } from 'vitest'
import { exportController } from './export-controller'

const mockApiGet = vi.fn()
const mockApiPost = vi.fn()

vi.mock('./http-client', () => ({
  apiGet: (...args: any[]) => mockApiGet(...args),
  apiPost: (...args: any[]) => mockApiPost(...args),
}))

describe('exportController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('exportModel calls POST /models/export with data', async () => {
    mockApiPost.mockResolvedValueOnce({ status: 'exported', format: 'sou', files: ['model.sou'] })
    const result = await exportController.exportModel({ format: 'sou', output_path: 'models/exported' })
    expect(mockApiPost).toHaveBeenCalledWith('/models/export', { format: 'sou', output_path: 'models/exported', include_tokenizer: true })
    expect(result.status).toBe('exported')
    expect(result.files).toContain('model.sou')
  })

  it('exportModel defaults output_path and include_tokenizer', async () => {
    mockApiPost.mockResolvedValueOnce({ status: 'exported', format: 'onnx', files: ['model.onnx'] })
    await exportController.exportModel({ format: 'onnx' })
    expect(mockApiPost).toHaveBeenCalledWith('/models/export', {
      format: 'onnx',
      output_path: 'models/exported',
      include_tokenizer: true,
    })
  })

  it('exportModel returns error when backend fails', async () => {
    mockApiPost.mockResolvedValueOnce({ status: 'error', error: 'No model loaded' })
    const result = await exportController.exportModel({ format: 'sou' })
    expect(result.error).toBe('No model loaded')
  })

  it('getFormats calls GET /models/export/formats', async () => {
    mockApiGet.mockResolvedValueOnce({ formats: ['sou', 'onnx', 'gguf'] })
    const formats = await exportController.getFormats()
    expect(mockApiGet).toHaveBeenCalledWith('/models/export/formats')
    expect(formats).toEqual(['sou', 'onnx', 'gguf'])
  })

  it('getFormats returns empty array on error', async () => {
    mockApiGet.mockRejectedValueOnce(new Error('Network error'))
    await expect(exportController.getFormats()).rejects.toThrow()
  })
})
