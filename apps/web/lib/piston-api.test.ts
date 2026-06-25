import { describe, expect, it, vi } from 'vitest'

const { mockPost, mockGet } = vi.hoisted(() => ({
  mockPost: vi.fn(),
  mockGet: vi.fn(),
}))

vi.mock('./http-client', () => ({
  createApiClient: () => ({
    post: mockPost,
    get: mockGet,
  }),
}))

import { executeCode, getPistonRuntimes } from './piston-api'

describe('executeCode', () => {
  it('returns stdout on success', async () => {
    mockPost.mockResolvedValue({
      data: { run: { stdout: 'hello\n', stderr: '', output: 'hello', code: 0 } },
    })
    const result = await executeCode('print("hello")')
    expect(result.output).toBe('hello')
  })

  it('returns stderr when no stdout', async () => {
    mockPost.mockResolvedValue({
      data: { run: { stdout: '', stderr: 'error line', output: '', code: 1 } },
    })
    const result = await executeCode('x = 1/0')
    expect(result.error).toBe('error line')
    expect(result.output).toBe('')
  })

  it('returns fallback message when no output at all', async () => {
    mockPost.mockResolvedValue({ data: { run: { code: 0 } } })
    const result = await executeCode('')
    expect(result.output).toBe('Code executed successfully (no output)')
  })

  it('handles HTTP error and returns error message', async () => {
    mockPost.mockRejectedValue(new Error('Network Error'))
    const result = await executeCode('print(1)')
    expect(result.error).toBe('HTTP Network Error')
  })

  it('passes language and version', async () => {
    mockPost.mockResolvedValue({ data: { run: { stdout: 'ok', stderr: '', output: 'ok', code: 0 } } })
    await executeCode('print(1)', 'javascript', '18')
    expect(mockPost).toHaveBeenCalledWith('/execute', {
      language: 'javascript',
      version: '18',
      files: [{ content: 'print(1)' }],
    })
  })
})

describe('getPistonRuntimes', () => {
  it('returns runtimes on success', async () => {
    mockGet.mockResolvedValue({ data: [{ language: 'python', version: '3.10', aliases: ['py'] }] })
    const result = await getPistonRuntimes()
    expect(result).toHaveLength(1)
    expect(result[0].language).toBe('python')
  })

  it('returns empty array on error', async () => {
    mockGet.mockRejectedValue(new Error('fail'))
    const result = await getPistonRuntimes()
    expect(result).toEqual([])
  })
})
