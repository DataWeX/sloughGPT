import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: vi.fn().mockResolvedValue([]),
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', model_type: null, model_loaded: false }),
    load: vi.fn(),
    unload: vi.fn(),
    getCacheUsage: vi.fn().mockResolvedValue({ used_gb: 0, limit_gb: 0, model_count: 0 }),
  },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: vi.fn().mockResolvedValue({ souls: [], current_soul: null }),
    switchSoul: vi.fn(),
    getCurrentSoul: vi.fn().mockResolvedValue(null),
    saveTraitWeights: vi.fn(),
    getTraitWeights: vi.fn().mockResolvedValue({}),
    listWeightSnapshots: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    runBenchmark: vi.fn(),
    getHistory: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({ healthLegacy: null }),
}))

vi.mock('@/lib/query/api-hooks', () => ({
  useModels: () => ({ data: [], isLoading: false, refetch: vi.fn() }),
  useSouls: () => ({ data: { souls: [], current_soul: null }, isLoading: false, refetch: vi.fn() }),
  useCurrentSoul: () => ({ data: null, refetch: vi.fn() }),
  useCheckpoints: () => ({ data: { checkpoints: [], active_checkpoint: null }, isLoading: false, refetch: vi.fn() }),
  useSwitchSoul: () => ({ mutateAsync: vi.fn() }),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: vi.fn().mockReturnValue('Error'),
}))

vi.mock('@/lib/inference-display', () => ({
  modelDisplayName: (id: string) => id,
}))

vi.mock('next/dynamic', () => {
  const React = require('react')
  return {
    __esModule: true,
    default: () => (props: Record<string, unknown>) => React.createElement('div', { 'data-testid': 'dynamic' }),
  }
})

vi.mock('@/components/models/ModelStatusCard', () => ({
  default: () => <div data-testid="model-status" />,
}))

vi.mock('@/components/models/ComposableLayersCard', () => ({
  default: () => <div data-testid="composable-layers" />,
}))

vi.mock('@/components/models/PersonalitiesCard', () => ({
  default: () => <div data-testid="personalities" />,
}))

vi.mock('@/components/models/PersonalityProfileCard', () => ({
  default: () => <div data-testid="personality-profile" />,
}))

vi.mock('@/components/models/ModelCatalogCard', () => ({
  default: () => <div data-testid="model-catalog" />,
}))

vi.mock('@/components/training/FineTunedModelsCard', () => ({
  FineTunedModelsCard: () => <div data-testid="finetuned-models" />,
}))

vi.mock('@/components/models/ModelPlaygroundCard', () => ({
  default: () => <div data-testid="model-playground" />,
}))

vi.mock('@/components/models/ModelCacheCard', () => ({
  default: () => <div data-testid="model-cache" />,
}))

vi.mock('@/components/models/QuantizationCard', () => ({
  default: () => <div data-testid="quantization" />,
}))

vi.mock('@/components/compare/ModelsCard', () => ({
  default: () => <div data-testid="compare-models" />,
}))

vi.mock('@/components/compare/ComparisonTableCard', () => ({
  default: () => <div data-testid="comparison-table" />,
}))

vi.mock('@/components/compare/SummaryCard', () => ({
  default: () => <div data-testid="summary" />,
}))

import ModelsPage from './page'

describe('ModelsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing', async () => {
    render(<ModelsPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('shows page header', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByText('Models & Personalities').length).toBeGreaterThanOrEqual(1)
  })

  it('shows connecting subtitle when health is null', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByText('Connecting...').length).toBeGreaterThanOrEqual(1)
  })

  it('renders model catalog card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('model-catalog').length).toBeGreaterThanOrEqual(1)
  })

  it('renders model cache card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('model-cache').length).toBeGreaterThanOrEqual(1)
  })

  it('renders model status card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('model-status').length).toBeGreaterThanOrEqual(1)
  })

  it('renders personalities card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('personalities').length).toBeGreaterThanOrEqual(1)
  })

  it('renders personality profile card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('personality-profile').length).toBeGreaterThanOrEqual(1)
  })

  it('renders finetuned models card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('finetuned-models').length).toBeGreaterThanOrEqual(1)
  })

  it('renders quantization card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('quantization').length).toBeGreaterThanOrEqual(1)
  })

  it('renders composable layers card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('composable-layers').length).toBeGreaterThanOrEqual(1)
  })

  it('renders model playground card', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('model-playground').length).toBeGreaterThanOrEqual(1)
  })

  it('renders refresh button', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByText('Refresh').length).toBeGreaterThanOrEqual(1)
  })

  it('renders all main sections', async () => {
    render(<ModelsPage />)
    await act(async () => {})
    expect(screen.getAllByTestId('model-catalog').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('model-cache').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('model-status').length).toBeGreaterThanOrEqual(1)
  })
})
