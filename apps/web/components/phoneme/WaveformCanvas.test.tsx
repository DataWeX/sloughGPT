import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/phoneme',
}))

afterEach(cleanup)

// Minimal valid WAV base64 (44-byte header + 2 samples of silence)
const SILENT_WAV = 'UklGRiYAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='

describe('WaveformCanvas', () => {
  it('renders with aria label', async () => {
    const { default: WaveformCanvas } = await import('./WaveformCanvas')
    render(<WaveformCanvas audioSrc={`data:audio/wav;base64,${SILENT_WAV}`} />)
    expect(screen.getByRole('img', { name: /audio waveform visualization/i })).toBeInTheDocument()
  })

  it('renders axis labels', async () => {
    const { default: WaveformCanvas } = await import('./WaveformCanvas')
    render(<WaveformCanvas audioSrc={`data:audio/wav;base64,${SILENT_WAV}`} />)
    expect(screen.getByText('Amplitude')).toBeInTheDocument()
    expect(screen.getByText('Time')).toBeInTheDocument()
  })

  it('renders canvas element', async () => {
    const { default: WaveformCanvas } = await import('./WaveformCanvas')
    render(<WaveformCanvas audioSrc={`data:audio/wav;base64,${SILENT_WAV}`} />)
    const canvas = screen.getByRole('img', { name: /audio waveform visualization/i })
    expect(canvas.tagName).toBe('CANVAS')
  })

  it('handles empty audio source', async () => {
    const { default: WaveformCanvas } = await import('./WaveformCanvas')
    render(<WaveformCanvas audioSrc="" />)
    expect(screen.getByRole('img', { name: /audio waveform visualization/i })).toBeInTheDocument()
  })

  it('handles invalid base64 audio gracefully', async () => {
    const { default: WaveformCanvas } = await import('./WaveformCanvas')
    render(<WaveformCanvas audioSrc="data:audio/wav;base64,invalid" />)
    expect(screen.getByRole('img', { name: /audio waveform visualization/i })).toBeInTheDocument()
  })
})
