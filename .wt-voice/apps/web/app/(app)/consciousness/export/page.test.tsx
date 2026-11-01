/**
 * Tests for the Consciousness Export page.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessExportPage from './page'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    getPersonality: vi.fn(),
    getStatus: vi.fn(),
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: 'en',
  }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

import { consciousnessController } from '@/lib/consciousness-controller'

const mockController = vi.mocked(consciousnessController)

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <LocaleProvider>{children}</LocaleProvider>
)

describe('ConsciousnessExportPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockController.getEpisodeHistory.mockResolvedValue({ episodes: [], total: 0 })
    mockController.getQualiaHistory.mockResolvedValue({ history: [] })
    mockController.getBeliefsHistory.mockResolvedValue({ beliefs: [] })
    mockController.getPersonality.mockResolvedValue({ traits: {}, description: '', name: '' })
    mockController.getStatus.mockResolvedValue({ enabled: true, level: 1, episodes: 0, beliefs: {} } as any)
    localStorage.clear()
  })

  it('renders the page', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.page_title').length).toBeGreaterThan(0)
    })
  })

  it('displays export options section', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.export_options_title').length).toBeGreaterThan(0)
    })
  })

  it('displays all export checkboxes', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.option_episodes').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_export.option_qualia').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_export.option_beliefs').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_export.option_personality').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_export.option_configuration').length).toBeGreaterThan(0)
    })
  })

  it('displays format selector', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.format_label').length).toBeGreaterThan(0)
      expect(screen.getAllByText('JSON').length).toBeGreaterThan(0)
      expect(screen.getAllByText('JSONL').length).toBeGreaterThan(0)
      expect(screen.getAllByText('CSV').length).toBeGreaterThan(0)
    })
  })

  it('displays export button', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.export_button').length).toBeGreaterThan(0)
    })
  })

  it('displays import section', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.import_title').length).toBeGreaterThan(0)
    })
  })

  it('displays import button', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.import_button').length).toBeGreaterThan(0)
    })
  })

  it('displays backups section', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.backups_title').length).toBeGreaterThan(0)
    })
  })

  it('shows no backups message initially', async () => {
    render(<ConsciousnessExportPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_export.no_backups').length).toBeGreaterThan(0)
    })
  })
})
