import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Progress: ({ value, size, variant, label, showValue }: any) => (
      <div data-testid="progress" data-value={value} data-variant={variant}>{label}</div>
    ),
  }
})
vi.mock('@/lib/query/api-hooks', () => ({
  useLoadModel: () => ({ mutateAsync: vi.fn().mockResolvedValue({ model_id: 'test-model' }) }),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))
vi.mock('@/lib/generate-controller', () => ({
  generateController: { generate: vi.fn().mockResolvedValue({}) },
}))
vi.mock('@/hooks/useConversionStatus', () => ({
  useConversionStatus: () => ({ status: null }),
  formatStage: (s: string) => s,
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn() },
}))
vi.mock('@/lib/inference-display', () => ({
  catalogIdMatchesRuntime: (a: string, b: string) => a === b,
}))

import ModelCatalogCard from './ModelCatalogCard'

const models = [
  { id: 'gpt2', name: 'GPT-2', source: 'huggingface', params: '124M', size_gb: 0.5, cached: true },
  { id: 'qwen', name: 'Qwen 0.5B', source: 'huggingface', params: '500M', size_gb: 1.0 },
  { id: 'custom', name: 'My Model', source: 'local', params: '10M' },
  { id: 'tagged', name: 'Tagged Model', tags: ['chat', 'small'] },
]

describe('ModelCatalogCard', () => {
  afterEach(cleanup)

  it('renders card title', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('Model Catalog')).toBeDefined()
  })

  it('shows skeleton when loading', () => {
    const { container } = render(<ModelCatalogCard models={[]} modelsLoading={true} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no models', () => {
    render(<ModelCatalogCard models={[]} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('No models available')).toBeDefined()
  })

  it('renders model names', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('Qwen 0.5B')).toBeDefined()
    expect(screen.getByText('My Model')).toBeDefined()
  })

  it('shows Load button for non-local models', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    const loadBtns = screen.getAllByText('Load')
    expect(loadBtns.length).toBeGreaterThanOrEqual(1)
  })

  it('shows "Local" label for local models', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('Local')).toBeDefined()
  })

  it('shows "Loaded" badge for active model', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId="gpt2" onModelLoaded={vi.fn()} />)
    const loadedEls = screen.getAllByText('Loaded')
    expect(loadedEls.length).toBeGreaterThanOrEqual(1)
  })

  it('shows "Cached" badge for cached models', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId="qwen" onModelLoaded={vi.fn()} />)
    expect(screen.getByText('Cached')).toBeDefined()
  })

  it('shows search input when more than 3 models', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByPlaceholderText('Search models...')).toBeDefined()
  })

  it('hides search when 3 or fewer models', () => {
    render(<ModelCatalogCard models={models.slice(0, 2)} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.queryByPlaceholderText('Search models...')).toBeNull()
  })

  it('filters models by search', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search models...'), { target: { value: 'qwen' } })
    expect(screen.getByText('Qwen 0.5B')).toBeDefined()
    expect(screen.queryByText('GPT-2')).toBeNull()
  })

  it('shows no-results message for empty search', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    fireEvent.change(screen.getByPlaceholderText('Search models...'), { target: { value: 'zzz' } })
    expect(screen.getByText(/No models matching/)).toBeDefined()
  })

  it('shows tags on model cards', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('chat')).toBeDefined()
    expect(screen.getByText('small')).toBeDefined()
  })

  it('shows model size', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    expect(screen.getByText('0.5 GB')).toBeDefined()
  })

  it('shows source badge for non-local', () => {
    render(<ModelCatalogCard models={models} modelsLoading={false} activeRuntimeId={null} onModelLoaded={vi.fn()} />)
    const sources = screen.getAllByText('huggingface')
    expect(sources.length).toBeGreaterThanOrEqual(1)
  })

  it('calls onModelLoaded after load completes', async () => {
    const onModelLoaded = vi.fn()
    render(<ModelCatalogCard models={[models[0]]} modelsLoading={false} activeRuntimeId={null} onModelLoaded={onModelLoaded} />)
    fireEvent.click(screen.getByText('Load'))
    await vi.waitFor(() => {
      expect(onModelLoaded).toHaveBeenCalled()
    })
  })
})
