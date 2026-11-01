// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { QuantizeCard } from './QuantizeCard'

const { mockQuantize, mockDequantize, mockAddToast } = vi.hoisted(() => ({
  mockQuantize: vi.fn(),
  mockDequantize: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    quantize: mockQuantize,
    dequantize: mockDequantize,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
})

describe('QuantizeCard', () => {
  it('returns null when model is not loaded', () => {
    const { container } = render(
      <QuantizeCard isLoaded={false} modelId="gpt2" health={null} />,
    )
    expect(container.querySelector('[data-testid="quantize-card"]')).toBeNull()
  })

  it('renders card when model is loaded', () => {
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    expect(screen.getAllByTestId('quantize-card').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Quantize').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Int8 and Int4 buttons', () => {
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    expect(screen.getAllByText('Int8').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Int4').length).toBeGreaterThanOrEqual(1)
  })

  it('calls quantize on Int8 click', async () => {
    mockQuantize.mockResolvedValue({ bits: 8, summary: { tensors: 50, avg_cosine_sim: 0.98, min_cosine_sim: 0.95, bits: 8 } })
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    fireEvent.click(screen.getByText('Int8'))
    await waitFor(() => expect(mockQuantize).toHaveBeenCalledWith(8, 'symmetric'))
    expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('8-bit'), 'success')
  })

  it('calls quantize on Int4 click', async () => {
    mockQuantize.mockResolvedValue({ bits: 4, summary: { tensors: 50, bits: 4, avg_cosine_sim: 0, min_cosine_sim: 0 } })
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    fireEvent.click(screen.getByText('Int4'))
    await waitFor(() => expect(mockQuantize).toHaveBeenCalledWith(4, 'symmetric'))
  })

  it('shows Restore button when quantized', () => {
    render(
      <QuantizeCard
        isLoaded={true}
        modelId="gpt2"
        health={{ quantization: { quantized: true, bits: 8, mode: 'symmetric', summary: { bits: 8, tensors: 50, avg_cosine_sim: 0.98, min_cosine_sim: 0.95 } } }}
      />,
    )
    expect(screen.getAllByText('Restore').length).toBeGreaterThanOrEqual(1)
  })

  it('calls dequantize on Restore click', async () => {
    mockDequantize.mockResolvedValue({ dequantized: true, model_type: 'gpt2', layers_reset: 12 })
    render(
      <QuantizeCard
        isLoaded={true}
        modelId="gpt2"
        health={{ quantization: { quantized: true, bits: 8 } }}
      />,
    )
    fireEvent.click(screen.getByText('Restore'))
    await waitFor(() => expect(mockDequantize).toHaveBeenCalled())
    expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('full precision'), 'success')
  })

  it('shows precision details when quantized', () => {
    render(
      <QuantizeCard
        isLoaded={true}
        modelId="gpt2"
        health={{ quantization: { quantized: true, bits: 8, mode: 'symmetric', summary: { bits: 8, tensors: 50, avg_cosine_sim: 0.98, min_cosine_sim: 0.95 } } }}
      />,
    )
    expect(screen.getAllByText('8-bit symmetric').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.980').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.950').length).toBeGreaterThanOrEqual(1)
  })

  it('shows placeholder text when not quantized', () => {
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    expect(screen.getAllByText(/reduce memory/).length).toBeGreaterThanOrEqual(1)
  })

  it('disables buttons during quantization', async () => {
    mockQuantize.mockImplementation(() => new Promise(() => {})) // never resolves
    render(
      <QuantizeCard isLoaded={true} modelId="gpt2" health={null} />,
    )
    fireEvent.click(screen.getByText('Int8'))
    await waitFor(() => {
      expect(screen.getAllByText('Working…').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows 8-bit badge when quantized', () => {
    render(
      <QuantizeCard
        isLoaded={true}
        modelId="gpt2"
        health={{ quantization: { quantized: true, bits: 8 } }}
      />,
    )
    expect(screen.getAllByText('8-bit').length).toBeGreaterThanOrEqual(1)
  })
})
