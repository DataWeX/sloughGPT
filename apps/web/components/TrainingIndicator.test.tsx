// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingIndicator } from './TrainingIndicator'
import { useTrainingSession } from '@/hooks/useTrainingSession'

vi.mock('@/hooks/useTrainingSession', () => ({
  useTrainingSession: vi.fn(),
}))

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

describe('TrainingIndicator', () => {
  const mockUseTrainingSession = vi.mocked(useTrainingSession)

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders nothing when idle', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: false,
      phase: 'idle',
      progress: 0,
      loss: null,
      method: null,
    } as any)
    const { container } = render(<TrainingIndicator />)
    expect(container.firstChild).toBeNull()
  })

  it('shows training status when running', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: true,
      phase: 'TRAINING',
      progress: 45,
      loss: 2.5,
      method: 'native',
    } as any)
    render(<TrainingIndicator />)
    expect(screen.getByText('Training')).toBeTruthy()
    expect(screen.getByText('45%')).toBeTruthy()
    expect(screen.getByText('loss 2.500')).toBeTruthy()
  })

  it('shows turbo training status', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: true,
      phase: 'TRAINING',
      progress: 75,
      loss: 1.2,
      method: 'turbo',
    } as any)
    render(<TrainingIndicator />)
    expect(screen.getByText('Turbo')).toBeTruthy()
    expect(screen.getByText('75%')).toBeTruthy()
  })

  it('shows complete status', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: false,
      phase: 'complete',
      progress: 100,
      loss: 0.5,
      method: 'native',
    } as any)
    render(<TrainingIndicator />)
    expect(screen.getByText('Training complete')).toBeTruthy()
  })

  it('shows error status', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: false,
      phase: 'error',
      progress: 0,
      loss: null,
      method: 'native',
    } as any)
    render(<TrainingIndicator />)
    expect(screen.getByText('Training failed')).toBeTruthy()
  })

  it('links to training page', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: true,
      phase: 'TRAINING',
      progress: 50,
      loss: null,
      method: 'native',
    } as any)
    const { unmount } = render(<TrainingIndicator />)
    const links = screen.getAllByText('Training')
    const link = links[0].closest('a')
    expect(link).toBeTruthy()
    expect(link?.getAttribute('href')).toBe('/training')
    unmount()
  })
})
