// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import ProviderDiagnosticsCard from './ProviderDiagnosticsCard'

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    debugProviders: vi.fn(),
  },
}))

import { modelController } from '@/lib/model-controller'

const mockData = {
  providers: {
    'default': {
      type: 'ProviderRouter',
      module: 'domains.models.provider',
      text_provider: 'slonet-native',
      processors: ['VisionProcessor', 'ToolUseProcessor'],
    },
    'slonet-native': {
      type: 'SloNetChatProvider',
      module: 'domains.inference.slonet_provider',
      model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
      server: { type: 'SloNetServer', has_circuit_breaker: true },
    },
  },
  default_provider: 'default',
  model_state: {
    model: 'SloNetChatProvider',
    model_type: 'Qwen/Qwen2.5-0.5B-Instruct',
    tokenizer: 'PreTrainedTokenizerFast',
    provider: 'SloNetChatProvider',
  },
  startup_phase: 'ready',
}

describe('ProviderDiagnosticsCard', () => {
  afterEach(cleanup)
  beforeEach(() => { vi.clearAllMocks() })

  it('renders provider chain on load', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('Provider Chain')).toBeDefined()
      expect(screen.getAllByText('ProviderRouter').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('SloNetChatProvider').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows model state details', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('Model State')).toBeDefined()
      expect(screen.getAllByText('Qwen/Qwen2.5-0.5B-Instruct').length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText('PreTrainedTokenizerFast')).toBeDefined()
    })
  })

  it('shows text provider name', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getAllByText(/slonet-native/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows all registered providers count', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/All Providers/)).toBeDefined()
    })
  })

  it('shows startup phase', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/Startup phase:/)).toBeDefined()
    })
  })

  it('shows error state', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(null)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('No response from server')).toBeDefined()
    })
  })

  it('shows error on fetch failure', async () => {
    vi.mocked(modelController.debugProviders).mockRejectedValue(new Error('network error'))
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/network error/)).toBeDefined()
    })
  })

  it('shows refresh button', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue(mockData)
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('Refresh')).toBeDefined()
    })
  })

  it('shows empty providers message', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue({
      providers: {},
      default_provider: null,
      model_state: { model: null, model_type: null, tokenizer: null, provider: null },
      startup_phase: 'loading',
    })
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText('No providers registered')).toBeDefined()
      expect(screen.getByText('No default router registered')).toBeDefined()
    })
  })

  it('shows broken chat warning when text_provider is NONE', async () => {
    vi.mocked(modelController.debugProviders).mockResolvedValue({
      providers: {
        'default': {
          type: 'ProviderRouter',
          module: 'domains.models.provider',
          processors: [],
        },
      },
      default_provider: 'default',
      model_state: { model: 'SloNetChatProvider', model_type: 'gpt2', tokenizer: 'TokenizerFast', provider: 'SloNetChatProvider' },
      startup_phase: 'ready',
    })
    render(<ProviderDiagnosticsCard />)
    await vi.waitFor(() => {
      expect(screen.getByText(/NONE — chat broken/)).toBeDefined()
    })
  })
})
