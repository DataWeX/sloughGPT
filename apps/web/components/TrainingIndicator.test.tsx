// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TrainingIndicator } from './TrainingIndicator'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { notify } from '@/lib/notify'

vi.mock('@/hooks/useTrainingSession', () => ({
  useTrainingSession: vi.fn(),
}))

vi.mock('@/vite/next-compat/link', () => ({
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}))

vi.mock('@/vite/next-compat/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/notify', () => ({
  notify: { toast: vi.fn(), banner: vi.fn() },
  useBannerStore: {
    getState: () => ({ banners: [], dismissBanner: vi.fn(), clearBanners: vi.fn() }),
  },
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

  it('routes failures to the global banner, not a sidebar label', () => {
    mockUseTrainingSession.mockReturnValue({
      trainingRunning: false,
      phase: 'error',
      progress: 0,
      loss: null,
      method: 'native',
      error: 'CUDA out of memory',
    } as any)
    const { container } = render(<TrainingIndicator />)
    // The sidebar row is silent on error — the banner is the failure surface.
    expect(container.firstChild).toBeNull()
    expect(screen.queryByText('Training failed')).toBeNull()
    expect(notify.banner).toHaveBeenCalledWith(
      expect.objectContaining({
        key: 'training-failed',
        tone: 'destructive',
        title: 'Training failed',
        message: 'CUDA out of memory',
        action: expect.objectContaining({ label: 'View training' }),
      }),
    )
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
