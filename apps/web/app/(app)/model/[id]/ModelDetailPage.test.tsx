// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const {
  mockPush, mockParams, mockModelList, mockGetHealth,
  mockModelLoad, mockUnload, mockBenchRun, mockAddToast,
} = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockParams: vi.fn(),
  mockModelList: vi.fn(),
  mockGetHealth: vi.fn(),
  mockModelLoad: vi.fn(),
  mockUnload: vi.fn(),
  mockBenchRun: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
  useParams: () => mockParams(),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: mockModelList,
    getHealth: mockGetHealth,
    load: mockModelLoad,
    unloadModel: mockUnload,
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: { run: mockBenchRun },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

import ModelDetailPage from './page'

const mockModel = {
  id: 'gpt2',
  name: 'GPT-2',
  source: 'huggingface',
  size_gb: 0.48,
  tags: ['text-generation', 'causal-lm'],
  params: '124M',
}

const mockHealthLoaded: any = {
  status: 'healthy',
  model_loaded: true,
  model_type: 'gpt2',
  device: 'cpu',
  inference_count: 42,
  vocab_size: 50257,
  block_size: 1024,
  summary: 'gpt2 loaded on cpu',
}

const mockHealthUnloaded: any = {
  status: 'healthy',
  model_loaded: false,
  model_type: '',
  device: 'cpu',
  inference_count: 0,
  summary: 'no model loaded',
}

const mockBenchmarkResult = {
  model: 'gpt2',
  num_parameters: 124000000,
  memory_mb: 487,
  throughput_tokens_per_sec: 45.2,
  inference_time_ms: 250,
  latency_p50_ms: 240,
  latency_p95_ms: 310,
  latency_p99_ms: 380,
}

beforeEach(() => {
  vi.clearAllMocks()
  mockParams.mockReturnValue({ id: 'gpt2' })
  mockModelList.mockResolvedValue([mockModel])
  mockGetHealth.mockResolvedValue(mockHealthLoaded)
})

afterEach(cleanup)

describe('ModelDetailPage', () => {
  it('renders header with model name', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('GPT-2')).toBeDefined())
    expect(screen.getByText('← Models')).toBeDefined()
  })

  it('shows loading skeleton while fetching', () => {
    mockModelList.mockReturnValue(new Promise(() => {}))
    const { container } = render(<ModelDetailPage />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('shows Loaded badge with status card when model loaded', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Loaded')).toBeDefined())
    expect(screen.getByText('42')).toBeDefined()
    expect(screen.getByText('0.48 GB')).toBeDefined()
  })

  it('shows Inactive badge when no model loaded', async () => {
    mockGetHealth.mockResolvedValue(mockHealthUnloaded)
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Inactive')).toBeDefined())
  })

  it('renders Load model button when not loaded', async () => {
    mockGetHealth.mockResolvedValue(mockHealthUnloaded)
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Load model')).toBeDefined())
  })

  it('loads model on button click', async () => {
    mockGetHealth.mockResolvedValue(mockHealthUnloaded)
    mockModelLoad.mockResolvedValue({ status: 'ok', device: 'cpu' })
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Load model')).toBeDefined())
    fireEvent.click(screen.getByText('Load model'))
    await waitFor(() => expect(mockModelLoad).toHaveBeenCalledWith('gpt2'))
    expect(mockAddToast).toHaveBeenCalledWith(
      expect.stringContaining('Model ready'), 'success',
    )
  })

  it('unloads model on Remove click', async () => {
    mockUnload.mockResolvedValue({})
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Remove')).toBeDefined())
    fireEvent.click(screen.getByText('Remove'))
    await waitFor(() => expect(mockUnload).toHaveBeenCalledWith('gpt2'))
  })

  it('shows Metrics card with Run benchmark button', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Run benchmark')).toBeDefined())
  })

  it('renders benchmark results', async () => {
    mockBenchRun.mockResolvedValue(mockBenchmarkResult)
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Run benchmark')).toBeDefined())
    fireEvent.click(screen.getByText('Run benchmark'))
    await waitFor(() => expect(screen.getByText('124M')).toBeDefined())
    expect(screen.getByText('487 MB')).toBeDefined()
  })

  it('renders quick test card when model loaded', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Quick test')).toBeDefined())
    expect(screen.getByPlaceholderText('Type a prompt to test...')).toBeDefined()
  })

  it('shows Details card with tags and vocab', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => {
      expect(screen.getByText('50,257')).toBeDefined()
      expect(screen.getByText('1,024')).toBeDefined()
    })
  })

  it('shows not found for unknown model id', async () => {
    mockModelList.mockResolvedValue([])
    render(<ModelDetailPage />)
    await waitFor(() => {
      expect(screen.getByText(/not found/)).toBeDefined()
    })
  })

  it('renders Chat with this model button when loaded', async () => {
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Chat with this model')).toBeDefined())
  })

  it('handles load error gracefully', async () => {
    mockGetHealth.mockResolvedValue(mockHealthUnloaded)
    mockModelLoad.mockRejectedValue(new Error('OOM'))
    render(<ModelDetailPage />)
    await waitFor(() => expect(screen.getByText('Load model')).toBeDefined())
    fireEvent.click(screen.getByText('Load model'))
    await waitFor(() => expect(screen.getByText('Error')).toBeDefined())
  })
})
