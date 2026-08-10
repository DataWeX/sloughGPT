import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, renderHook, act } from '@testing-library/react'
import React from 'react'

const mockController = vi.hoisted(() => ({
  modelController: {
    list: vi.fn(),
    getHealth: vi.fn(),
    load: vi.fn(),
    loadModelPath: vi.fn(),
    unloadModel: vi.fn(),
  },
}))

vi.mock('@/lib/model-controller', () => mockController)

import { ModelProvider, useModels, useCurrentModel, useModelById, useLocalModels, useHuggingFaceModels } from './ModelContext'
import { liveStatusStore } from '@/hooks/useLiveStatus'

const mockModels = [
  { id: 'm1', name: 'GPT-2', type: 'huggingface', loaded: true, size_mb: 512, params: '124M', description: 'base', tags: ['gen'] },
  { id: 'm2', name: 'Local Net', type: 'local', loaded: false, size_mb: 8 },
  { id: 'm3', name: 'No Type' },
]

const healthyHealth = { status: 'healthy', model_loaded: true, model_type: 'gpt2', summary: 'ok', inference_count: 3, is_inferencing: false }

function wrapper({ children }: { children: React.ReactNode }) {
  return <ModelProvider>{children}</ModelProvider>
}

beforeEach(() => {
  liveStatusStore.getState().reset()
  vi.clearAllMocks()
})

afterEach(() => cleanup())

describe('ModelContext', () => {
  it('renders children inside the provider', () => {
    render(<ModelProvider><div>model child</div></ModelProvider>)
    expect(screen.getByText('model child')).toBeDefined()
  })

  it('useModels throws outside provider', () => {
    expect(() => renderHook(() => useModels())).toThrow('useModels must be used within a ModelProvider')
  })

  it('starts with empty models, no loading, no error', () => {
    const { result } = renderHook(() => useModels(), { wrapper })
    expect(result.current.models).toEqual([])
    expect(result.current.currentModel).toBeNull()
    expect(result.current.loading).toBe(false)
    expect(result.current.loadingModelId).toBeNull()
    expect(result.current.error).toBeNull()
    expect(result.current.isModelLoaded).toBe(false)
  })

  it('does not refresh models until the API is ready', async () => {
    renderHook(() => useModels(), { wrapper })
    await act(async () => {})
    expect(mockController.modelController.list).not.toHaveBeenCalled()
  })

  it('refreshes models once the API is ready and maps fields', async () => {
    mockController.modelController.list.mockResolvedValue(mockModels)
    const { result } = renderHook(() => useModels(), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    expect(mockController.modelController.list).toHaveBeenCalled()
    expect(result.current.models).toEqual([
      { id: 'm1', name: 'GPT-2', type: 'huggingface', loaded: true, size_mb: 512, size_gb: undefined, params: '124M', description: 'base', tags: ['gen'] },
      { id: 'm2', name: 'Local Net', type: 'local', loaded: false, size_mb: 8, size_gb: undefined, params: undefined, description: undefined, tags: undefined },
      { id: 'm3', name: 'No Type', type: 'huggingface', loaded: false, size_mb: undefined, size_gb: undefined, params: undefined, description: undefined, tags: undefined },
    ])
  })

  it('keeps models empty when list fails', async () => {
    mockController.modelController.list.mockRejectedValue(new Error('network'))
    const { result } = renderHook(() => useModels(), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    expect(result.current.models).toEqual([])
    expect(result.current.error).toBeNull()
  })

  it('loadModel succeeds and refreshes models + health', async () => {
    mockController.modelController.load.mockResolvedValue({ status: 'ok' })
    mockController.modelController.list.mockResolvedValue(mockModels)
    mockController.modelController.getHealth.mockResolvedValue(healthyHealth)
    const { result } = renderHook(() => useModels(), { wrapper })
    await act(async () => {
      const res = await result.current.loadModel('m1', { device: 'cpu' })
      expect(res).toEqual({ success: true })
    })
    expect(mockController.modelController.load).toHaveBeenCalledWith('m1', 'cpu')
    expect(mockController.modelController.list).toHaveBeenCalled()
    expect(mockController.modelController.getHealth).toHaveBeenCalled()
    expect(result.current.loading).toBe(false)
    expect(result.current.loadingModelId).toBeNull()
  })

  it('loadModel surfaces errors and returns failure', async () => {
    mockController.modelController.load.mockRejectedValue(new Error('OOM'))
    const { result } = renderHook(() => useModels(), { wrapper })
    let res: { success: boolean; error?: string } | undefined
    await act(async () => {
      res = await result.current.loadModel('m1')
    })
    expect(res).toEqual({ success: false, error: 'OOM' })
    expect(result.current.error).toBe('OOM')
    expect(result.current.loading).toBe(false)
  })

  it('loadModel reports backend error status instead of false success', async () => {
    mockController.modelController.load.mockResolvedValue({ status: 'error', error: 'No .slnc file for gpt2' })
    const { result } = renderHook(() => useModels(), { wrapper })
    let res: { success: boolean; error?: string } | undefined
    await act(async () => {
      res = await result.current.loadModel('gpt2')
    })
    expect(res).toEqual({ success: false, error: 'No .slnc file for gpt2' })
    expect(result.current.error).toBe('No .slnc file for gpt2')
    expect(mockController.modelController.list).not.toHaveBeenCalled()
    expect(mockController.modelController.getHealth).not.toHaveBeenCalled()
    expect(result.current.loading).toBe(false)
  })

  it('loadModelPath succeeds and refreshes', async () => {
    mockController.modelController.loadModelPath.mockResolvedValue({ status: 'ok' })
    mockController.modelController.list.mockResolvedValue([])
    mockController.modelController.getHealth.mockResolvedValue(healthyHealth)
    const { result } = renderHook(() => useModels(), { wrapper })
    await act(async () => {
      const res = await result.current.loadModelPath('/tmp/model.sou')
      expect(res).toEqual({ success: true })
    })
    expect(mockController.modelController.loadModelPath).toHaveBeenCalledWith('/tmp/model.sou')
    expect(result.current.loading).toBe(false)
  })

  it('loadModelPath reports backend error field', async () => {
    mockController.modelController.loadModelPath.mockResolvedValue({ status: 'error', error: 'bad path' })
    const { result } = renderHook(() => useModels(), { wrapper })
    let res: { success: boolean; error?: string } | undefined
    await act(async () => {
      res = await result.current.loadModelPath('/nope')
    })
    expect(res).toEqual({ success: false, error: 'bad path' })
    expect(result.current.error).toBe('bad path')
  })

  it('unloadModel succeeds and refreshes', async () => {
    mockController.modelController.unloadModel.mockResolvedValue({ status: 'ok' })
    mockController.modelController.list.mockResolvedValue([])
    mockController.modelController.getHealth.mockResolvedValue(healthyHealth)
    const { result } = renderHook(() => useModels(), { wrapper })
    await act(async () => {
      const res = await result.current.unloadModel('m1')
      expect(res).toEqual({ success: true })
    })
    expect(mockController.modelController.unloadModel).toHaveBeenCalledWith('m1')
  })

  it('unloadModel surfaces errors', async () => {
    mockController.modelController.unloadModel.mockRejectedValue(new Error('busy'))
    const { result } = renderHook(() => useModels(), { wrapper })
    let res: { success: boolean; error?: string } | undefined
    await act(async () => {
      res = await result.current.unloadModel('m1')
    })
    expect(res).toEqual({ success: false, error: 'busy' })
    expect(result.current.error).toBe('busy')
  })

  it('clearError resets the error', async () => {
    mockController.modelController.load.mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => useModels(), { wrapper })
    await act(async () => { await result.current.loadModel('m1') })
    expect(result.current.error).toBe('boom')
    act(() => result.current.clearError())
    expect(result.current.error).toBeNull()
  })

  it('syncs live health into isModelLoaded and currentModel when connected', async () => {
    const { result } = renderHook(() => useModels(), { wrapper })
    expect(result.current.isModelLoaded).toBe(false)
    act(() => {
      liveStatusStore.getState().setHealth({
        model_loaded: true,
        model_loading: false,
        model_type: 'gpt2',
        is_inferencing: false,
        inference_count: 3,
        health_status: 'healthy',
        health_summary: 'ok',
      } as never)
      liveStatusStore.getState().setConnectionStatus('connected')
    })
    expect(result.current.isModelLoaded).toBe(true)
    expect(result.current.currentModel).toBe('gpt2')
  })

  it('does not sync live health when disconnected', () => {
    const { result } = renderHook(() => useModels(), { wrapper })
    act(() => {
      liveStatusStore.getState().setHealth({ model_loaded: true, model_type: 'gpt2' } as never)
      liveStatusStore.getState().setConnectionStatus('offline')
    })
    expect(result.current.isModelLoaded).toBe(false)
  })

  it('useCurrentModel reflects model id and loaded state', async () => {
    mockController.modelController.list.mockResolvedValue(mockModels)
    const { result } = renderHook(() => useCurrentModel(), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    act(() => {
      liveStatusStore.getState().setHealth({ model_loaded: true, model_type: 'gpt2' } as never)
      liveStatusStore.getState().setConnectionStatus('connected')
    })
    expect(result.current.modelId).toBe('gpt2')
    expect(result.current.isLoaded).toBe(true)
  })

  it('useModelById finds a model', async () => {
    mockController.modelController.list.mockResolvedValue(mockModels)
    const { result } = renderHook(() => useModelById('m2'), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    expect(result.current?.name).toBe('Local Net')
  })

  it('useModelById returns undefined for missing id', async () => {
    mockController.modelController.list.mockResolvedValue(mockModels)
    const { result } = renderHook(() => useModelById('nope'), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    expect(result.current).toBeUndefined()
  })

  it('useLocalModels and useHuggingFaceModels filter by type', async () => {
    mockController.modelController.list.mockResolvedValue(mockModels)
    const local = renderHook(() => useLocalModels(), { wrapper })
    const hf = renderHook(() => useHuggingFaceModels(), { wrapper })
    act(() => liveStatusStore.getState().setReady(true))
    await act(async () => {})
    expect(local.result.current.map(m => m.id)).toEqual(['m2'])
    expect(hf.result.current.map(m => m.id)).toEqual(['m1', 'm3'])
  })
})
