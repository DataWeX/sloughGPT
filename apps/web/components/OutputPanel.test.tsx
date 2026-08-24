import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

const mockHookReturn = vi.fn()

vi.mock('@/hooks/useServerOutput', () => ({
  useServerOutput: () => mockHookReturn(),
}))

afterEach(cleanup)

function defaultHookReturn(overrides: Record<string, any> = {}) {
  return {
    lines: [
      { ts: 1700000000, level: 'info', source: 'system', text: 'Server started' },
      { ts: 1700000001, level: 'error', source: 'model', text: 'CUDA OOM' },
      { ts: 1700000002, level: 'info', source: 'api', text: 'GET /health' },
    ],
    streaming: true,
    clear: vi.fn(),
    scrollRef: { current: null },
    paused: false,
    togglePause: vi.fn(),
    exportLines: vi.fn(),
    ...overrides,
  }
}

describe('OutputPanel', () => {
  it('returns null when not open', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    const { container } = render(<OutputPanel open={false} onClose={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders when open', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('Service Output').length).toBeGreaterThanOrEqual(1)
  })

  it('shows line count', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('(3/3)').length).toBeGreaterThanOrEqual(1)
  })

  it('renders log line text', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('Server started').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('CUDA OOM').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onClose', async () => {
    const onClose = vi.fn()
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={onClose} />)
    fireEvent.click(screen.getAllByLabelText('Close')[0])
    expect(onClose).toHaveBeenCalled()
  })

  it('toggles pause', async () => {
    const togglePause = vi.fn()
    mockHookReturn.mockReturnValue(defaultHookReturn({ togglePause }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.click(screen.getAllByLabelText('Pause output')[0])
    expect(togglePause).toHaveBeenCalled()
  })

  it('shows resume when paused', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn({ paused: true }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByLabelText('Resume output').length).toBeGreaterThanOrEqual(1)
  })

  it('calls clear', async () => {
    const clear = vi.fn()
    mockHookReturn.mockReturnValue(defaultHookReturn({ clear }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.click(screen.getAllByText('Clear')[0])
    expect(clear).toHaveBeenCalled()
  })

  it('calls exportLines', async () => {
    const exportLines = vi.fn()
    mockHookReturn.mockReturnValue(defaultHookReturn({ exportLines }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.click(screen.getAllByLabelText('Export as log file')[0])
    expect(exportLines).toHaveBeenCalledWith('text')
  })

  it('shows empty state when no lines', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn({ lines: [], streaming: false }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('No output yet').length).toBeGreaterThanOrEqual(1)
  })

  it('filters by level toggle', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    const errorBtn = screen.getAllByText('error')[0]
    fireEvent.click(errorBtn)
    expect(screen.getAllByText('CUDA OOM').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('Server started')).not.toBeInTheDocument()
  })

  it('filters by search text', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.change(screen.getAllByLabelText('Search output')[0], { target: { value: 'CUDA' } })
    expect(screen.getAllByText('CUDA OOM').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('Server started')).not.toBeInTheDocument()
  })

  it('filters by source', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.change(screen.getAllByLabelText('Filter by source')[0], { target: { value: 'system' } })
    expect(screen.getAllByText('Server started').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('CUDA OOM')).not.toBeInTheDocument()
  })

  it('shows waiting for output', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn({ lines: [], streaming: true }))
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    expect(screen.getAllByText('Waiting for output...').length).toBeGreaterThanOrEqual(1)
  })

  it('shows no matching lines', async () => {
    mockHookReturn.mockReturnValue(defaultHookReturn())
    const { OutputPanel } = await import('@/components/OutputPanel')
    render(<OutputPanel open={true} onClose={vi.fn()} />)
    fireEvent.change(screen.getAllByLabelText('Search output')[0], { target: { value: 'zzz nonexistent' } })
    expect(screen.getAllByText('No matching lines').length).toBeGreaterThanOrEqual(1)
  })
})
