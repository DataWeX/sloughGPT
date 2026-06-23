// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/link', () => ({ default: ({ children, href }: any) => <a href={href}>{children}</a> }))

import { InferenceStatusBar, InferenceRuntimeToolbar, ModelStatusBar } from './InferenceStatusBar'

const healthOk = { status: 'ok', model_loaded: true, model_type: 'gpt2', num_parameters: 124_000_000, vocab_size: 50257, block_size: 1024, inference_count: 42 } as any

describe('InferenceStatusBar', () => {
  afterEach(cleanup)

  it('returns null when health is ok and model loaded and catalog matches', () => {
    const { container } = render(<InferenceStatusBar health={healthOk} selectedCatalogId="gpt2" />)
    expect(container.innerHTML).toBe('')
  })

  it('shows offline message when health is offline', () => {
    render(<InferenceStatusBar health={'offline' as any} selectedCatalogId="gpt2" />)
    expect(screen.getByText('API unreachable')).toBeDefined()
  })

  it('shows no weights message when model not loaded', () => {
    const h = { status: 'ok', model_loaded: false, model_type: '' } as any
    render(<InferenceStatusBar health={h} selectedCatalogId="gpt2" />)
    expect(screen.getByText('No weights loaded')).toBeDefined()
  })

  it('shows mismatch when catalog does not match runtime', () => {
    render(<InferenceStatusBar health={healthOk} selectedCatalogId="tinyllama" />)
    expect(screen.getByText(/Catalog ≠ runtime/)).toBeDefined()
  })

  it('renders with role="status"', () => {
    render(<InferenceStatusBar health={'offline' as any} selectedCatalogId="gpt2" />)
    expect(screen.getByRole('status')).toBeDefined()
  })
})

describe('InferenceRuntimeToolbar', () => {
  afterEach(cleanup)

  it('shows ... when null', () => {
    render(<InferenceRuntimeToolbar health={null as any} onRefresh={vi.fn()} />)
    expect(screen.getByText('...')).toBeDefined()
  })

  it('shows Offline when offline', () => {
    render(<InferenceRuntimeToolbar health={'offline' as any} onRefresh={vi.fn()} />)
    expect(screen.getByText('Offline')).toBeDefined()
  })

  it('shows model type when loaded', () => {
    render(<InferenceRuntimeToolbar health={healthOk} onRefresh={vi.fn()} />)
    expect(screen.getByText('gpt2')).toBeDefined()
  })

  it('shows No model when not loaded', () => {
    const h = { status: 'ok', model_loaded: false, model_type: '' } as any
    render(<InferenceRuntimeToolbar health={h} onRefresh={vi.fn()} />)
    expect(screen.getByText('No model')).toBeDefined()
  })
})

describe('ModelStatusBar', () => {
  afterEach(cleanup)

  it('renders with loading state when null', () => {
    const { container } = render(<ModelStatusBar health={null as any} />)
    expect(container.querySelector('[role="status"]')).toBeDefined()
  })

  it('renders with offline state', () => {
    render(<ModelStatusBar health={'offline' as any} />)
    expect(screen.getByLabelText('API offline')).toBeDefined()
  })

  it('renders with loaded state and is clickable', () => {
    render(<ModelStatusBar health={healthOk} />)
    expect(screen.getByLabelText('Model loaded: gpt2')).toBeDefined()
  })

  it('renders no-model state', () => {
    const h = { status: 'ok', model_loaded: false, model_type: '' } as any
    render(<ModelStatusBar health={h} />)
    expect(screen.getByLabelText('No model loaded')).toBeDefined()
  })

  it('opens dialog on click when loaded', () => {
    render(<ModelStatusBar health={healthOk} />)
    fireEvent.click(screen.getByLabelText('Model loaded: gpt2'))
    expect(screen.getByText('Model Details')).toBeDefined()
  })

  it('shows model parameters in dialog', () => {
    render(<ModelStatusBar health={healthOk} />)
    fireEvent.click(screen.getByLabelText('Model loaded: gpt2'))
    expect(screen.getByText('0.1B')).toBeDefined()
  })

  it('shows vocab size in dialog', () => {
    render(<ModelStatusBar health={healthOk} />)
    fireEvent.click(screen.getByLabelText('Model loaded: gpt2'))
    expect(screen.getByText('50,257')).toBeDefined()
  })

  it('shows block size in dialog', () => {
    render(<ModelStatusBar health={healthOk} />)
    fireEvent.click(screen.getByLabelText('Model loaded: gpt2'))
    expect(screen.getByText('1024')).toBeDefined()
  })

  it('shows inference count in dialog', () => {
    render(<ModelStatusBar health={healthOk} />)
    fireEvent.click(screen.getByLabelText('Model loaded: gpt2'))
    expect(screen.getByText('42')).toBeDefined()
  })
})
