import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/phoneme',
}))

afterEach(cleanup)

const mockSpectrogram = {
  data: [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7],
  ],
  n_mels: 3,
  n_frames: 5,
}

describe('SpectrogramCanvas', () => {
  it('renders with aria label', async () => {
    const { default: SpectrogramCanvas } = await import('./SpectrogramCanvas')
    render(<SpectrogramCanvas spectrogram={mockSpectrogram} />)
    expect(screen.getByRole('img', { name: /mel spectrogram visualization/i })).toBeInTheDocument()
  })

  it('renders frequency labels', async () => {
    const { default: SpectrogramCanvas } = await import('./SpectrogramCanvas')
    render(<SpectrogramCanvas spectrogram={mockSpectrogram} />)
    expect(screen.getByText('Low frequency')).toBeInTheDocument()
    expect(screen.getByText('High frequency')).toBeInTheDocument()
  })

  it('renders time labels', async () => {
    const { default: SpectrogramCanvas } = await import('./SpectrogramCanvas')
    render(<SpectrogramCanvas spectrogram={mockSpectrogram} />)
    expect(screen.getByText('Start')).toBeInTheDocument()
    expect(screen.getByText('End')).toBeInTheDocument()
  })

  it('renders canvas element', async () => {
    const { default: SpectrogramCanvas } = await import('./SpectrogramCanvas')
    render(<SpectrogramCanvas spectrogram={mockSpectrogram} />)
    const canvas = screen.getByRole('img', { name: /mel spectrogram visualization/i })
    expect(canvas.tagName).toBe('CANVAS')
  })

  it('handles empty spectrogram gracefully', async () => {
    const { default: SpectrogramCanvas } = await import('./SpectrogramCanvas')
    render(
      <SpectrogramCanvas
        spectrogram={{ data: [], n_mels: 0, n_frames: 0 }}
      />
    )
    expect(screen.getByRole('img', { name: /mel spectrogram visualization/i })).toBeInTheDocument()
  })
})
