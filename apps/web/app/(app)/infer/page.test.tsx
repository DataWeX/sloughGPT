import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGenerate = vi.fn()
const mockHealth = vi.fn()
const mockInfo = vi.fn()
const mockEmbed = vi.fn()
const mockTokenize = vi.fn()
const mockDetokenize = vi.fn()
const mockAddToast = vi.fn()

vi.mock('@/lib/infer-controller', () => ({
  inferController: {
    generate: (...args: unknown[]) => mockGenerate(...args),
    health: (...args: unknown[]) => mockHealth(...args),
    info: (...args: unknown[]) => mockInfo(...args),
    embed: (...args: unknown[]) => mockEmbed(...args),
    tokenize: (...args: unknown[]) => mockTokenize(...args),
    detokenize: (...args: unknown[]) => mockDetokenize(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, type, step, className, placeholder }: any) => (
      <input value={value} onChange={onChange} type={type} step={step} className={className} placeholder={placeholder} />
    ),
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
    Textarea: ({ value, onChange, rows, className, placeholder }: any) => (
      <textarea value={value} onChange={onChange} rows={rows} className={className} placeholder={placeholder} />
    ),
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import InferPage from './page'

describe('InferPage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: true })
    render(<InferPage />)
    expect(screen.getByText('API Playground')).toBeInTheDocument()
    expect(screen.getByText('Test inference endpoints directly')).toBeInTheDocument()
  })

  it('fetches health on mount', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    render(<InferPage />)
    await waitFor(() => {
      expect(mockHealth).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('shows health status', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('shows unreachable on health failure', async () => {
    mockHealth.mockRejectedValue(new Error('fail'))
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('unreachable')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('renders tab buttons', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: true })
    render(<InferPage />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
    expect(screen.getByText('Embed')).toBeInTheDocument()
    expect(screen.getByText('Tokenize')).toBeInTheDocument()
    expect(screen.getByText('Model Info')).toBeInTheDocument()
  })

  it('calls generate on Run generate click', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    mockGenerate.mockResolvedValue({ text: 'Hello world', model: 'gpt2', tokens_generated: 2, elapsed_ms: 50 })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Run generate'))
    await waitFor(() => {
      expect(mockGenerate).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('shows generate result', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    mockGenerate.mockResolvedValue({ text: 'Generated text', model: 'gpt2', tokens_generated: 5, elapsed_ms: 100 })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Run generate'))
    await waitFor(() => {
      expect(screen.getByText('Generated text')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('5 tokens')).toBeInTheDocument()
    expect(screen.getByText('100ms')).toBeInTheDocument()
  })

  it('shows error toast on generate failure', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    mockGenerate.mockRejectedValue(new Error('OOM'))
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Run generate'))
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Generation failed: OOM', 'error')
    }, { timeout: 5000 })
  })

  it('switches to embed tab and calls embed', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: true })
    mockEmbed.mockResolvedValue({ embedding: [0.1, 0.2], dimensions: 2, model: 'e5-small' })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Embed'))
    fireEvent.click(screen.getByText('Run embed'))
    await waitFor(() => {
      expect(mockEmbed).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('switches to tokenize tab and calls tokenize', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    mockTokenize.mockResolvedValue({ tokens: ['hello', 'world'], ids: [1, 2], count: 2 })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Tokenize'))
    fireEvent.click(screen.getByText('Run tokenize'))
    await waitFor(() => {
      expect(mockTokenize).toHaveBeenCalled()
    }, { timeout: 5000 })
    await waitFor(() => {
      expect(screen.getByText('Tokens (2)')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('switches to info tab and loads model info', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: true, has_streaming: true, has_embedding: false })
    mockInfo.mockResolvedValue({ model_id: 'gpt2', model_type: 'gpt2', num_parameters: 124000000, vocab_size: 50257, max_context: 1024, num_layers: 12, has_tokenizer: true, has_streaming: true, has_embedding: false, extra: {} })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Model Info'))
    fireEvent.click(screen.getByText('Load Info'))
    await waitFor(() => {
      expect(screen.getByText('Model Info')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('disables button when model not loaded', async () => {
    mockHealth.mockResolvedValue({ status: 'ok', model_loaded: false, has_streaming: false, has_embedding: false })
    render(<InferPage />)
    await waitFor(() => {
      expect(screen.getByText('ok')).toBeInTheDocument()
    }, { timeout: 5000 })
    const btn = screen.getByRole('button', { name: /Run generate/i })
    expect(btn).toBeDisabled()
  })
})
