import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessTestingPage from './page'
import { consciousnessController } from '@/lib/consciousness-controller'

const mockController = vi.hoisted(() => ({
  getStatus: vi.fn().mockResolvedValue({}),
  getSelfModel: vi.fn().mockResolvedValue({}),
  getQualia: vi.fn().mockResolvedValue({}),
  reflect: vi.fn().mockResolvedValue({}),
  updateConfig: vi.fn().mockResolvedValue({}),
  getTrainingStatus: vi.fn().mockResolvedValue({}),
  startTraining: vi.fn().mockResolvedValue({}),
  evaluate: vi.fn().mockResolvedValue({}),
  getEpisodeHistory: vi.fn().mockResolvedValue({ episodes: [], total: 0 }),
  getQualiaHistory: vi.fn().mockResolvedValue({ history: [] }),
  getBeliefsHistory: vi.fn().mockResolvedValue({ beliefs: [] }),
  submitFeedback: vi.fn().mockResolvedValue({}),
  seedData: vi.fn().mockResolvedValue({}),
  getPersonality: vi.fn().mockResolvedValue({}),
  updatePersonality: vi.fn().mockResolvedValue({}),
  resetPersonality: vi.fn().mockResolvedValue({}),
  clearEpisodes: vi.fn().mockResolvedValue({}),
  resetBeliefs: vi.fn().mockResolvedValue({}),
  getPersonalityHistory: vi.fn().mockResolvedValue({}),
  getPersonalityPresets: vi.fn().mockResolvedValue({}),
  applyPersonalityPreset: vi.fn().mockResolvedValue({}),
  getPersonalityConflicts: vi.fn().mockResolvedValue({}),
  listPersonas: vi.fn().mockResolvedValue({}),
  savePersona: vi.fn().mockResolvedValue({}),
  getPersona: vi.fn().mockResolvedValue({}),
  activatePersona: vi.fn().mockResolvedValue({}),
  deletePersona: vi.fn().mockResolvedValue({}),
  healthCheck: vi.fn().mockResolvedValue({}),
  backup: vi.fn().mockResolvedValue({}),
  restore: vi.fn().mockResolvedValue({}),
  downloadBackup: vi.fn().mockResolvedValue({}),
  importBackup: vi.fn().mockResolvedValue({}),
  getStats: vi.fn().mockResolvedValue({}),
  batch: vi.fn().mockResolvedValue({}),
  connectStream: vi.fn().mockReturnValue(vi.fn()),
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: mockController,
}))

vi.mock('@/hooks/useLocale', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/useLocale')>()
  return {
    ...actual,
    useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  }
})

vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(vi.fn((sel: any) => sel({ addToast: vi.fn() })), { getState: () => ({ addToast: vi.fn() }) }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers() }))

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessTestingPage />
    </LocaleProvider>
  )
}

describe('ConsciousnessTestingPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows API tester section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.api_tester_title').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint select with options', async () => {
    renderPage()
    await waitFor(() => {
      const select = document.querySelector('select')
      expect(select).toBeTruthy()
      const options = select!.querySelectorAll('option')
      expect(options.length).toBe(9)
    })
  })

  it('shows Send button for API tester', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.send_request').length).toBeGreaterThan(0)
    })
  })

  it('shows manual reflect section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.manual_title').length).toBeGreaterThan(0)
    })
  })

  it('shows batch runner section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.batch_title').length).toBeGreaterThan(0)
    })
  })

  it('shows state inspector section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.state_title').length).toBeGreaterThan(0)
    })
  })

  it('shows reset button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.reset_defaults').length).toBeGreaterThan(0)
    })
  })

  it('shows seed button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_testing.seed_50').length).toBeGreaterThan(0)
    })
  })

  it('can change endpoint selection', async () => {
    renderPage()
    await waitFor(() => {
      const select = document.querySelector('select') as HTMLSelectElement
      expect(select).toBeTruthy()
      fireEvent.change(select, { target: { value: '2' } })
      expect(select.value).toBe('2')
    })
  })
})
