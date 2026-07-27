import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', async () => {
  const actual = await vi.importActual<typeof import('@sloughgpt/strui')>('@sloughgpt/strui')
  return {
    ...actual,
    Badge: ({ label, variant, size }: any) => <span data-testid="badge" data-variant={variant}>{label}</span>,
    Skeleton: ({ className }: any) => <div data-testid="skeleton" className={className} />,
  }
})

import ModelsCard from './ModelsCard'

const models = [
  { id: 'gpt2', name: 'GPT-2', loaded: true, sizeGb: 0.5, source: 'huggingface' },
  { id: 'qwen', name: 'Qwen', loaded: false, sizeGb: 1.2, source: 'huggingface', type: 'text-generation' },
  { id: 'custom', name: 'Custom Model', loaded: true, source: 'local' },
]

describe('ModelsCard', () => {
  afterEach(cleanup)

  it('renders card title', () => {
    render(<ModelsCard models={[]} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('Models')).toBeDefined()
  })

  it('shows skeleton when loading', () => {
    const { container } = render(<ModelsCard models={[]} loading={true} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    const skeletons = container.querySelectorAll('[data-testid="skeleton"]')
    expect(skeletons.length).toBe(4)
  })

  it('shows empty state when no models', () => {
    render(<ModelsCard models={[]} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText(/No models available/)).toBeDefined()
  })

  it('renders model cards', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('GPT-2')).toBeDefined()
    expect(screen.getByText('Qwen')).toBeDefined()
    expect(screen.getByText('Custom Model')).toBeDefined()
  })

  it('shows Loaded badge for loaded models', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    const badges = screen.getAllByTestId('badge')
    const loadedBadge = badges.find(b => b.textContent === 'Loaded')
    expect(loadedBadge).toBeDefined()
  })

  it('shows size in GB when available', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('0.5 GB')).toBeDefined()
    expect(screen.getByText('1.2 GB')).toBeDefined()
  })

  it('shows source badge', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    const sources = screen.getAllByText('huggingface')
    expect(sources.length).toBeGreaterThanOrEqual(1)
  })

  it('shows Benchmark button for each model', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    const btns = screen.getAllByText('Benchmark')
    expect(btns.length).toBe(3)
  })

  it('calls onBenchmark when Benchmark clicked', () => {
    const onBenchmark = vi.fn()
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set()} onBenchmark={onBenchmark} onClear={vi.fn()} />)
    fireEvent.click(screen.getAllByText('Benchmark')[0])
    expect(onBenchmark).toHaveBeenCalledWith('gpt2')
  })

  it('shows Rerun button when result exists', () => {
    const results = { gpt2: { model: 'gpt2', latency_ms: 100, throughput: 10, num_parameters: 124000000, memory_mb: 500, throughput_tokens_per_sec: 10, inference_time_ms: 100 } }
    render(<ModelsCard models={models} loading={false} results={results} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('Rerun')).toBeDefined()
  })

  it('shows Benchmarking... when running', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set(['gpt2'])} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('Benchmarking…')).toBeDefined()
  })

  it('disables Benchmark button when running', () => {
    render(<ModelsCard models={models} loading={false} results={{}} running={new Set(['gpt2'])} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    const btn = screen.getByText('Benchmarking…').closest('button')!
    expect(btn.disabled).toBe(true)
  })

  it('shows Trash button when result exists', () => {
    const results = { gpt2: { model: 'gpt2', latency_ms: 100, throughput: 10, num_parameters: 124000000, memory_mb: 500, throughput_tokens_per_sec: 10, inference_time_ms: 100 } }
    render(<ModelsCard models={models} loading={false} results={results} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByLabelText('Clear result for gpt2')).toBeDefined()
  })

  it('calls onClear when Trash clicked', () => {
    const onClear = vi.fn()
    const results = { gpt2: { model: 'gpt2', latency_ms: 100, throughput: 10, num_parameters: 124000000, memory_mb: 500, throughput_tokens_per_sec: 10, inference_time_ms: 100 } }
    render(<ModelsCard models={models} loading={false} results={results} running={new Set()} onBenchmark={vi.fn()} onClear={onClear} />)
    fireEvent.click(screen.getByLabelText('Clear result for gpt2'))
    expect(onClear).toHaveBeenCalledWith('gpt2')
  })

  it('shows error when result has error', () => {
    const results = { gpt2: { model: 'gpt2', latency_ms: 0, throughput: 0, num_parameters: 0, memory_mb: 0, throughput_tokens_per_sec: 0, inference_time_ms: 0, error: 'OOM' } }
    render(<ModelsCard models={models} loading={false} results={results} running={new Set()} onBenchmark={vi.fn()} onClear={vi.fn()} />)
    expect(screen.getByText('OOM')).toBeDefined()
  })
})
