import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessTestRunnerPage from './page'

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
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

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
  ok: true,
  status: 200,
  text: () => Promise.resolve(JSON.stringify({ data: {} })),
}))

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <LocaleProvider>{children}</LocaleProvider>
)

describe('ConsciousnessTestRunnerPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ data: {} })),
    }))
  })

  it('renders without crashing', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_test_runner.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows test suites', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_test_runner.suites_title').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Core Engine Tests').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Qualia Tests').length).toBeGreaterThan(0)
      expect(screen.getAllByText('API Tests').length).toBeGreaterThan(0)
    })
  })

  it('shows run all button', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_test_runner.run_all').length).toBeGreaterThan(0)
    })
  })

  it('shows run selected button', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_test_runner.run_selected').length).toBeGreaterThan(0)
    })
  })

  it('shows suite checkboxes', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('Self-Model Tests').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Personality Tests').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Integration Tests').length).toBeGreaterThan(0)
    })
  })

  it('does not show results before running', async () => {
    render(<ConsciousnessTestRunnerPage />, { wrapper: TestWrapper })
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_test_runner.page_title').length).toBeGreaterThan(0)
    })
    expect(screen.queryByText('consciousness_test_runner.results_title')).not.toBeInTheDocument()
  })
})
