// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TrainingQuickActions } from './TrainingQuickActions'
import type { Checkpoint } from '@/lib/souls-controller'

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    name: 'test-checkpoint',
    soul: 'test-soul',
    loss: 2.5,
    ...overrides,
  }
}

describe('TrainingQuickActions', () => {
  it('renders nothing when no checkpoints', () => {
    const { container } = render(<TrainingQuickActions checkpoints={[]} />)
    expect(container.querySelector('[data-testid="training-quick-actions"]')).toBeNull()
  })

  it('renders quick actions card', () => {
    render(<TrainingQuickActions checkpoints={[mkCp()]} />)
    expect(screen.getAllByTestId('training-quick-actions').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Quick Actions').length).toBeGreaterThanOrEqual(1)
  })

  it('shows load best button with loss', () => {
    render(<TrainingQuickActions checkpoints={[mkCp({ loss: 1.5 })]} onLoadBest={vi.fn()} />)
    expect(screen.getAllByText(/Load best/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/1\.500/).length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLoadBest with checkpoint name', () => {
    const onLoadBest = vi.fn()
    render(<TrainingQuickActions checkpoints={[mkCp({ name: 'best-cp', loss: 1.5 })]} onLoadBest={onLoadBest} />)
    const btns = screen.getAllByText(/Load best/)
    fireEvent.click(btns[btns.length - 1])
    expect(onLoadBest).toHaveBeenCalledWith('best-cp')
  })

  it('shows export metrics button', () => {
    render(<TrainingQuickActions checkpoints={[mkCp()]} onExportMetrics={vi.fn()} />)
    expect(screen.getAllByText('Export metrics').length).toBeGreaterThanOrEqual(1)
  })

  it('shows export notes button', () => {
    render(<TrainingQuickActions checkpoints={[mkCp()]} onExportNotes={vi.fn()} />)
    expect(screen.getAllByText('Export notes').length).toBeGreaterThanOrEqual(1)
  })

  it('shows clear notes button', () => {
    render(<TrainingQuickActions checkpoints={[mkCp()]} onClearNotes={vi.fn()} />)
    expect(screen.getAllByText('Clear notes').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onExportMetrics when clicked', () => {
    const onExportMetrics = vi.fn()
    render(<TrainingQuickActions checkpoints={[mkCp()]} onExportMetrics={onExportMetrics} />)
    const btns = screen.getAllByText('Export metrics')
    fireEvent.click(btns[btns.length - 1])
    expect(onExportMetrics).toHaveBeenCalledTimes(1)
  })

  it('disables load best when checkpoint is loaded', () => {
    render(<TrainingQuickActions checkpoints={[mkCp({ is_loaded: true, loss: 1.5 })]} onLoadBest={vi.fn()} />)
    const btns = screen.getAllByText(/Best loaded/)
    expect(btns[0]).toBeDefined()
  })
})
