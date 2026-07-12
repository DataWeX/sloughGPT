// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import QuantizationCard from './QuantizationCard'
import { modelController, type QuantizationResult } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    quantize: vi.fn(),
  },
  ...{ QuantizationResult: {} as new () => QuantizationResult },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn(() => vi.fn()),
}))

const mockResult: QuantizationResult = {
  quantized: true,
  bits: 8,
  mode: 'symmetric',
  model_type: 'slonet',
  layers_quantized: 15,
  total_layers: 15,
  summary: { tensors: 15, bits: 8, avg_cosine_sim: 0.9987, min_cosine_sim: 0.9921 },
  per_tensor: {},
  avx2_enabled: true,
}

const mockResultWithLayers: QuantizationResult = {
  ...mockResult,
  per_tensor: {
    'blocks.0.q_proj.weight': { scale: 0.01, zero_point: 0, cosine_sim: 0.9995 },
    'blocks.0.k_proj.weight': { scale: 0.02, zero_point: 0, cosine_sim: 0.9980 },
    'blocks.0.fc1.weight': { scale: 0.03, zero_point: 0, cosine_sim: 0.9920 },
    'lm_head.weight': { scale: 0.04, zero_point: 0, cosine_sim: 0.9970 },
  },
}

describe('QuantizationCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('shows offline message when not online', () => {
    render(<QuantizationCard isOnline={false} />)
    expect(screen.getByText(/Load a model first/)).toBeDefined()
  })

  it('shows precision selector and Apply button when online', () => {
    render(<QuantizationCard isOnline={true} />)
    expect(screen.getByText('int8')).toBeDefined()
    expect(screen.getByText('int4')).toBeDefined()
    expect(screen.getByText('Apply')).toBeDefined()
  })

  it('highlights int8 by default', () => {
    render(<QuantizationCard isOnline={true} />)
    const int8Btn = screen.getByText('int8')
    expect(int8Btn.className).toContain('bg-primary')
  })

  it('toggles to int4 on click', () => {
    render(<QuantizationCard isOnline={true} />)
    fireEvent.click(screen.getByText('int4'))
    const int4Btn = screen.getByText('int4')
    expect(int4Btn.className).toContain('bg-primary')
  })

  it('calls modelController.quantize on Apply', async () => {
    const mockQuantize = vi.mocked(modelController.quantize)
    mockQuantize.mockResolvedValue(mockResult)

    render(<QuantizationCard isOnline={true} />)
    fireEvent.click(screen.getByText('Apply'))

    await waitFor(() => {
      expect(mockQuantize).toHaveBeenCalledWith(8, 'symmetric')
    })
  })

  it('shows results after quantization', async () => {
    const mockQuantize = vi.mocked(modelController.quantize)
    mockQuantize.mockResolvedValue(mockResult)

    render(<QuantizationCard isOnline={true} />)
    fireEvent.click(screen.getByText('Apply'))

    await waitFor(() => {
      expect(screen.getByText('15/15')).toBeDefined()
      expect(screen.getByText('0.9987')).toBeDefined()
      expect(screen.getByText('Enabled')).toBeDefined()
    })
  })

  it('shows error toast on failure', async () => {
    const addToast = vi.fn()
    vi.mocked(useToastStore).mockReturnValue(addToast)
    const mockQuantize = vi.mocked(modelController.quantize)
    mockQuantize.mockRejectedValue(new Error('No model loaded'))

    render(<QuantizationCard isOnline={true} />)
    fireEvent.click(screen.getByText('Apply'))

    await waitFor(() => {
      expect(addToast).toHaveBeenCalledWith('No model loaded', 'error')
    })
  })

  it('expands per-layer detail table after quantization', async () => {
    const mockQuantize = vi.mocked(modelController.quantize)
    mockQuantize.mockResolvedValue(mockResultWithLayers)

    render(<QuantizationCard isOnline={true} />)
    fireEvent.click(screen.getByText('Apply'))

    await waitFor(() => {
      expect(screen.getByText(/Show.*per-layer detail/)).toBeDefined()
    })

    fireEvent.click(screen.getByText(/Show.*per-layer detail/))

    await waitFor(() => {
      expect(screen.getByText('B0/Q')).toBeDefined()
      expect(screen.getByText('B0/FC1')).toBeDefined()
      expect(screen.getByText('lm_head')).toBeDefined()
      expect(screen.getByText(/Hide.*per-layer detail/)).toBeDefined()
    })
  })
})
