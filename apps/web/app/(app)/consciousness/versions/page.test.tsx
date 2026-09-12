import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessVersionsPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: vi.fn(),
    getPersonality: vi.fn(),
    updateConfig: vi.fn(),
    updatePersonality: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: 'en',
  }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

import { consciousnessController } from '@/lib/consciousness-controller'

const mockController = vi.mocked(consciousnessController)

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <LocaleProvider>{children}</LocaleProvider>
)

describe('ConsciousnessVersionsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mockController.getStatus.mockResolvedValue({ level: 2, enabled: true } as any)
    mockController.getPersonality.mockResolvedValue({ voice: 'Analytical', traits: ['curious', 'precise'], style: 'Formal' } as any)
    mockController.updateConfig.mockResolvedValue({ updated: true } as any)
    mockController.updatePersonality.mockResolvedValue({ updated: true } as any)
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows version history section', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.history_title').length).toBeGreaterThan(0)
    })
  })

  it('shows save button', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.save_button').length).toBeGreaterThan(0)
    })
  })

  it('shows current config section', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.current_title').length).toBeGreaterThan(0)
    })
  })

  it('shows auto-save section', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.auto_save_title').length).toBeGreaterThan(0)
    })
  })

  it('displays no versions message when empty', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_versions.no_versions').length).toBeGreaterThan(0)
    })
  })

  it('shows current config values', async () => {
    render(<ConsciousnessVersionsPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Full').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Active').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Analytical').length).toBeGreaterThan(0)
    })
  })
})
