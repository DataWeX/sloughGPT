import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/hooks/useServerOutput', () => ({
  useServerOutput: vi.fn(() => ({
    lines: [
      { ts: 1700000000, level: 'info', source: 'server', text: 'Model loaded' },
      { ts: 1700000001, level: 'error', source: 'inference', text: 'OOM killed' },
    ],
    streaming: true,
    clear: vi.fn(),
    scrollRef: { current: null },
    paused: false,
    togglePause: vi.fn(),
    exportLines: vi.fn(),
  })),
}))

afterEach(cleanup)

describe('OutputCard', () => {
  it('renders title and live indicator', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByText('Server Output')).toBeInTheDocument()
    expect(screen.getByText('Live')).toBeInTheDocument()
  })

  it('renders log lines with correct text', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByText('Model loaded')).toBeInTheDocument()
    expect(screen.getByText('OOM killed')).toBeInTheDocument()
  })

  it('renders level labels', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByText('INF')).toBeInTheDocument()
    expect(screen.getByText('ERR')).toBeInTheDocument()
  })

  it('renders pause and export buttons', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByLabelText('Pause output')).toBeInTheDocument()
    expect(screen.getByLabelText('Export as log file')).toBeInTheDocument()
    expect(screen.getByLabelText('Clear output')).toBeInTheDocument()
  })

  it('shows empty state when no lines', async () => {
    const { useServerOutput } = await import('@/hooks/useServerOutput')
    vi.mocked(useServerOutput).mockReturnValueOnce({
      lines: [],
      streaming: false,
      clear: vi.fn(),
      scrollRef: { current: null },
      paused: false,
      togglePause: vi.fn(),
      exportLines: vi.fn(),
    })
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByText('Output will appear here during server activity')).toBeInTheDocument()
  })

  it('renders source labels', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    render(<OutputCard />)
    expect(screen.getByText('server')).toBeInTheDocument()
    expect(screen.getByText('inference')).toBeInTheDocument()
  })

  it('has scrollable container', async () => {
    const { OutputCard } = await import('@/components/OutputCard')
    const { container } = render(<OutputCard />)
    expect(container.querySelector('[class*="overflow"]')).toBeDefined()
  })
})
