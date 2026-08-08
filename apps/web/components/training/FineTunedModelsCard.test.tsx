import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const mockPush = vi.hoisted(() => vi.fn())
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listFineTuned: vi.fn(),
    loadFineTuned: vi.fn(),
    deleteFineTuned: vi.fn(),
  },
}))
vi.mock('@/lib/model-controller', () => ({
  modelController: {
    unloadModel: vi.fn().mockResolvedValue({ status: 'unloaded' }),
  },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

import { FineTunedModelsCard } from './FineTunedModelsCard'
import { trainingJobsController } from '@/lib/training-controller'
import { modelController } from '@/lib/model-controller'

const mocks = vi.mocked(trainingJobsController)

const mkModel = (overrides: Record<string, any> = {}) => ({
  name: 'gpt2_dataset_1', model: 'gpt2', model_name: 'gpt2_dataset_1', dataset: 'dataset_1', size_mb: 1.2,
  model_path: '/tmp/finetuned/gpt2_dataset_1', ...overrides,
})

describe('FineTunedModelsCard', () => {
  afterEach(() => { cleanup(); vi.clearAllMocks() })

  beforeEach(() => {
    mocks.listFineTuned.mockResolvedValue([])
  })

  it('renders title and empty state', async () => {
    render(<FineTunedModelsCard />)
    expect(screen.getByText('Fine-tuned models')).toBeDefined()
    await waitFor(() => {
      expect(screen.getByText(/No fine-tuned models yet/)).toBeDefined()
    })
  })

  it('lists fine-tuned models', async () => {
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    render(<FineTunedModelsCard />)
    await waitFor(() => {
      expect(screen.getByText('gpt2_dataset_1')).toBeDefined()
      expect(screen.getAllByText(/gpt2/).length).toBeGreaterThan(0)
      expect(screen.getByText(/1.2 MB/)).toBeDefined()
    })
  })

  it('shows final loss and epochs when present', async () => {
    mocks.listFineTuned.mockResolvedValue([mkModel({ final_loss: 0.4242, epochs: 3 })])
    render(<FineTunedModelsCard />)
    await waitFor(() => {
      expect(screen.getByText(/loss 0.4242/)).toBeDefined()
      expect(screen.getByText(/3 ep/)).toBeDefined()
    })
  })

  it('marks active model and hides its Load button', async () => {
    const m = mkModel()
    mocks.listFineTuned.mockResolvedValue([m])
    render(<FineTunedModelsCard activeModelId="gpt2_dataset_1" />)
    await waitFor(() => {
      expect(screen.getByText('gpt2_dataset_1')).toBeDefined()
    })
    expect(screen.queryByText('Load')).toBeNull()
  })

  it('marks active model by model_name id', async () => {
    const m = mkModel()
    mocks.listFineTuned.mockResolvedValue([m])
    render(<FineTunedModelsCard activeModelId="gpt2_dataset_1" />)
    await waitFor(() => {
      expect(screen.getByText('gpt2_dataset_1')).toBeDefined()
    })
    expect(screen.queryByText('Load')).toBeNull()
  })

  it('does not mark active when only base model id matches', async () => {
    const m = mkModel()
    mocks.listFineTuned.mockResolvedValue([m])
    render(<FineTunedModelsCard activeModelId="gpt2" />)
    await waitFor(() => {
      expect(screen.getByText('gpt2_dataset_1')).toBeDefined()
    })
    expect(screen.getByText('Load')).toBeDefined()
  })

  it('loads fine-tuned model on button click and calls onLoaded', async () => {
    const onLoaded = vi.fn()
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    mocks.loadFineTuned.mockResolvedValue({ status: 'loaded', name: 'gpt2_dataset_1', model_path: '/tmp/x' })
    render(<FineTunedModelsCard onLoaded={onLoaded} />)
    await waitFor(() => expect(screen.getByText('gpt2_dataset_1')).toBeDefined())
    fireEvent.click(screen.getByText('Load'))
    await waitFor(() => {
      expect(mocks.loadFineTuned).toHaveBeenCalledWith('gpt2_dataset_1')
      expect(onLoaded).toHaveBeenCalled()
    })
  })

  it('deletes fine-tuned model and refreshes list', async () => {
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    mocks.deleteFineTuned.mockResolvedValue({ status: 'deleted', name: 'gpt2_dataset_1' })
    render(<FineTunedModelsCard />)
    await waitFor(() => expect(screen.getByText('gpt2_dataset_1')).toBeDefined())
    fireEvent.click(screen.getByRole('button', { name: /Delete/ }))
    await waitFor(() => {
      expect(mocks.deleteFineTuned).toHaveBeenCalledWith('gpt2_dataset_1')
      expect(mocks.listFineTuned).toHaveBeenCalledTimes(2)
    })
  })

  it('shows Unload for the active model', async () => {
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    render(<FineTunedModelsCard activeModelId="gpt2_dataset_1" />)
    await waitFor(() => expect(screen.getByText('gpt2_dataset_1')).toBeDefined())
    expect(screen.getByRole('button', { name: /Unload/ })).toBeDefined()
  })

  it('unloads the active model and calls onLoaded', async () => {
    const onLoaded = vi.fn()
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    render(<FineTunedModelsCard activeModelId="gpt2_dataset_1" onLoaded={onLoaded} />)
    await waitFor(() => expect(screen.getByText('gpt2_dataset_1')).toBeDefined())
    fireEvent.click(screen.getByRole('button', { name: /Unload/ }))
    await waitFor(() => {
      expect(modelController.unloadModel).toHaveBeenCalled()
      expect(onLoaded).toHaveBeenCalled()
    })
  })

  it('navigates to model detail page on name click', async () => {
    mocks.listFineTuned.mockResolvedValue([mkModel()])
    render(<FineTunedModelsCard />)
    await waitFor(() => expect(screen.getByText('gpt2_dataset_1')).toBeDefined())
    fireEvent.click(screen.getByRole('button', { name: /View details/ }))
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith('/model/gpt2_dataset_1'))
  })
})