// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockModels: any = { models: [], loading: false, loadingModelId: null, loadModel: vi.fn(), currentModel: null, isModelLoaded: false }
const mockLocalModels: any[] = []
const mockHfModels: any[] = []

vi.mock('@/contexts/ModelContext', () => ({
  useModels: () => mockModels,
  useLocalModels: () => mockLocalModels,
  useHuggingFaceModels: () => mockHfModels,
}))

import { ModelSelector, ModelCard } from './ModelSelector'

describe('ModelSelector', () => {
  afterEach(cleanup)

  it('renders select with placeholder', () => {
    render(<ModelSelector />)
    expect(screen.getByText('Select a model')).toBeDefined()
  })

  it('shows Load button when value is set', () => {
    mockModels.currentModel = null
    render(<ModelSelector value="gpt2" />)
    expect(screen.getByText('Load')).toBeDefined()
  })

  it('shows loading state', () => {
    mockModels.loading = true
    mockModels.loadingModelId = 'gpt2'
    render(<ModelSelector value="gpt2" />)
    expect(screen.getByText('Loading...')).toBeDefined()
    mockModels.loading = false
    mockModels.loadingModelId = null
  })

  it('shows Loaded check when current model matches', () => {
    mockModels.currentModel = 'gpt2'
    mockModels.isModelLoaded = true
    render(<ModelSelector value="gpt2" />)
    expect(screen.getByText('Loaded ✓')).toBeDefined()
    mockModels.currentModel = null
    mockModels.isModelLoaded = false
  })
})

describe('ModelCard', () => {
  const model = { id: 'gpt2', name: 'GPT-2', type: 'local', sizeMb: 500 }

  afterEach(cleanup)

  it('renders model name and type', () => {
    render(<ModelCard model={model} />)
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('local')).toBeDefined()
  })

  it('shows Active when isActive', () => {
    render(<ModelCard model={model} isActive={true} />)
    expect(screen.getByText('Active')).toBeDefined()
  })

  it('shows Load button when onLoad provided and not active', () => {
    render(<ModelCard model={model} onLoad={vi.fn()} />)
    expect(screen.getByText('Load')).toBeDefined()
  })

  it('shows loading state', () => {
    render(<ModelCard model={model} isLoading={true} onLoad={vi.fn()} />)
    expect(screen.getByText('Loading...')).toBeDefined()
  })

  it('renders tags', () => {
    render(<ModelCard model={{ ...model, tags: ['transformer', 'text-generation'] }} />)
    expect(screen.getByText('transformer')).toBeDefined()
    expect(screen.getByText('text-generation')).toBeDefined()
  })

  it('renders description', () => {
    render(<ModelCard model={{ ...model, description: 'A small GPT model' }} />)
    expect(screen.getByText('A small GPT model')).toBeDefined()
  })

  it('renders size in MB', () => {
    render(<ModelCard model={model} />)
    expect(screen.getByText('500.0 MB')).toBeDefined()
  })

  it('renders size in KB when < 1 MB', () => {
    render(<ModelCard model={{ ...model, sizeMb: 0.5 }} />)
    expect(screen.getByText('512 KB')).toBeDefined()
  })
})
