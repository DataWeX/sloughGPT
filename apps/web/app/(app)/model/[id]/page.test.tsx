import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.join(' '),
    Button: ({ children, onClick, disabled, variant, size, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} aria-label={ariaLabel} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ label, variant, size, className, children }: any) => <span data-variant={variant} className={className}>{label || children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children, columns }: any) => <div data-columns={columns}>{children}</div>,
    KeyValueList: ({ items }: any) => <div>{items?.map((item: any, i: number) => <div key={i} data-testid={`kv-${item.label}`}><span>{item.label}</span><span>{item.value}</span></div>)}</div>,
    SettingsRow: ({ title, control, children }: any) => <div><span>{title}</span>{control || children}</div>,
    Slider: ({ value, onValueChange, min, max, step }: any) => (
      <input type="range" value={value?.[0]} min={min} max={max} step={step}
        onChange={e => onValueChange?.([Number(e.target.value)])} />
    ),
    Breadcrumbs: ({ items, className }: any) => <nav aria-label="Breadcrumb" className={className}>{items?.map((item: any, i: number) => <span key={i}>{item.label}</span>)}</nav>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconTrash: () => <span data-testid="icon-trash">trash</span>,
  }
})

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  addToast: vi.fn(),
  list: vi.fn(),
  getHealth: vi.fn(),
  load: vi.fn(),
  unloadModel: vi.fn(),
  listFineTuned: vi.fn(),
  loadFineTuned: vi.fn(),
  run: vi.fn(),
  get: vi.fn(),
  update: vi.fn(),
  apiGet: vi.fn(),
}))

const stableRouter = { push: vi.fn() }
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'gpt2' }), useRouter: () => stableRouter }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mocks.addToast }) }))
vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: mocks.list,
    getHealth: mocks.getHealth,
    load: mocks.load,
    unloadModel: mocks.unloadModel,
  },
}))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listFineTuned: mocks.listFineTuned,
    loadFineTuned: mocks.loadFineTuned,
  },
}))
vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: { run: mocks.run },
}))
vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: { get: mocks.get, update: mocks.update },
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any, fallback: string) => e instanceof Error ? e.message : fallback,
}))
vi.mock('@/lib/http-client', () => ({
  apiGet: mocks.apiGet,
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn() },
}))
vi.mock('@/components/model/QuantizeCard', () => ({
  QuantizeCard: ({ isLoaded, modelId }: any) => <div data-testid="quantize-card" data-loaded={isLoaded}>Quantize</div>,
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, children, loading, loadingContent }: any) => (
    <div className="sl-page mx-auto max-w-4xl">
      <h1>{loading ? '...' : title}</h1>
      {loading ? loadingContent : children}
    </div>
  ),
}))

import Page from './page'

const SAMPLE_MODEL: any = {
  id: 'gpt2', name: 'GPT-2', source: 'huggingface', description: 'Small model',
  size_gb: 0.5, params: '124M', type: 'text-generation', tags: ['gpt', 'small'],
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.list.mockResolvedValue([SAMPLE_MODEL])
  mocks.getHealth.mockResolvedValue({ model_loaded: false, device: null })
  mocks.listFineTuned.mockResolvedValue([])
  mocks.apiGet.mockResolvedValue({ logs: [] })
  mocks.get.mockResolvedValue({ temperature: 0.7, max_new_tokens: 256, top_p: 1.0, top_k: 50 })
  mocks.update.mockResolvedValue({})
})

afterEach(() => cleanup())

describe('ModelDetailPage', () => {
  it('renders loading state', async () => {
    mocks.list.mockImplementation(() => new Promise(() => {}))
    render(<Page />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model name and breadcrumbs', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('GPT-2').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByText('Models')).toBeTruthy()
  })

  it('shows status card with Loaded badge', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Loaded')).toBeTruthy()
    })
  })

  it('shows inactive status when not loaded', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Inactive')).toBeTruthy()
    })
  })

  it('shows load button when not loaded', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Load model')).toBeTruthy()
    })
  })

  it('loads model', async () => {
    mocks.load.mockResolvedValue({ device: 'cpu' })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Load model')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Load model'))
    await waitFor(() => {
      expect(mocks.load).toHaveBeenCalledWith('gpt2')
      expect(mocks.addToast).toHaveBeenCalledWith('Model ready: gpt2 (cpu)', 'success')
    })
  })

  it('shows error toast on load failure', async () => {
    mocks.load.mockRejectedValue(new Error('OOM'))
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Load model')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Load model'))
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('OOM', 'error')
    })
  })

  it('unloads model', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    mocks.unloadModel.mockResolvedValue({})
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Remove')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Remove'))
    await waitFor(() => {
      expect(mocks.unloadModel).toHaveBeenCalled()
      expect(mocks.addToast).toHaveBeenCalledWith('Model stopped', 'info')
    })
  })

  it('shows metrics card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Metrics')).toBeTruthy()
    })
    expect(screen.getByText('Load this model to see live metrics.')).toBeTruthy()
  })

  it('runs benchmark', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    mocks.run.mockResolvedValue({
      num_parameters: 124000000, memory_mb: 500, throughput_tokens_per_sec: 10,
      inference_time_ms: 50, latency_p50_ms: 45, latency_p95_ms: 55, latency_p99_ms: 60,
    })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Run benchmark')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Run benchmark'))
    await waitFor(() => {
      expect(screen.getByText('124.0M')).toBeTruthy()
      expect(screen.getByText('500 MB')).toBeTruthy()
    })
  })

  it('shows benchmark error', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    mocks.run.mockRejectedValue(new Error('timeout'))
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Run benchmark')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Run benchmark'))
    await waitFor(() => {
      expect(screen.getByText(/Benchmark failed/)).toBeTruthy()
    })
  })

  it('shows generation config card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Generation Config')).toBeTruthy()
    })
    expect(screen.getByText('Temperature')).toBeTruthy()
    expect(screen.getByText('Max tokens')).toBeTruthy()
    expect(screen.getByText('Top-p')).toBeTruthy()
    expect(screen.getByText('Top-k')).toBeTruthy()
  })

  it('saves generation config', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Generation Config')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Save'))
    await waitFor(() => {
      expect(mocks.update).toHaveBeenCalled()
      expect(mocks.addToast).toHaveBeenCalledWith('Generation config updated', 'success')
    })
  })

  it('shows quantize card when loaded', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByTestId('quantize-card')).toBeTruthy()
    })
  })

  it('hides quantize card when not loaded', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Load model')).toBeTruthy()
    })
    expect(screen.queryByTestId('quantize-card')).toBeNull()
  })

  it('shows details card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Details')).toBeTruthy()
    })
    expect(screen.getAllByText('Type').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model not found', async () => {
    mocks.list.mockResolvedValue([])
    mocks.listFineTuned.mockResolvedValue([])
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText(/not found/)).toBeTruthy()
    })
  })

  it('shows error toast on load failure', async () => {
    mocks.list.mockRejectedValue(new Error('network'))
    render(<Page />)
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Something went wrong loading the model', 'error')
    })
  })

  it('navigates to chat with model', async () => {
    mocks.getHealth.mockResolvedValue({ model_loaded: true, model_type: 'gpt2', device: 'cpu' })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Chat with this model')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Chat with this model'))
    expect(stableRouter.push).toHaveBeenCalledWith('/chat')
  })
})
