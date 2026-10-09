import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessApiExplorerPage from './page'

import { createMockController } from '@/lib/__test-helper'
vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: createMockController(),
}))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(vi.fn((sel: any) => sel({ addToast: vi.fn() })), { getState: () => ({ addToast: vi.fn() }) }),
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('@/components/icons/NavIcons', () => ({
  IconCode: (p: any) => <span data-testid="icon-code" {...p} />,
  IconPlus: (p: any) => <span data-testid="icon-plus" {...p} />,
  IconTrash: (p: any) => <span data-testid="icon-trash" {...p} />,
  IconClock: (p: any) => <span data-testid="icon-clock" {...p} />,
}))
vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers() }))

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessApiExplorerPage />
    </LocaleProvider>
  )
}

describe('ConsciousnessApiExplorerPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers() }))
    localStorage.clear()
  })

  it('renders page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows category list', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.cat_core').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_history').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_training').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_personality').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_personas').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_data').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_api_explorer.cat_advanced').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint list', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/status').length).toBeGreaterThan(0)
    })
  })

  it('shows GET method badge', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('GET').length).toBeGreaterThan(0)
    })
  })

  it('shows POST method badge', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('POST').length).toBeGreaterThan(0)
    })
  })

  it('shows Send button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.send_request').length).toBeGreaterThan(0)
    })
  })

  it('shows request body textarea', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/status').length).toBeGreaterThan(0)
    })
    const reflectBtn = screen.getAllByText('/consciousness/reflect')[0]
    fireEvent.click(reflectBtn)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.body').length).toBeGreaterThan(0)
    })
  })

  it('shows history section', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.history').length).toBeGreaterThan(0)
    })
  })

  it('category click changes endpoint list', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/status').length).toBeGreaterThan(0)
    })
    const trainingCategory = screen.getAllByText('consciousness_api_explorer.cat_training')[0]
    fireEvent.click(trainingCategory)
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/train/status').length).toBeGreaterThan(0)
    })
  })

  it('can type in request body', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/status').length).toBeGreaterThan(0)
    })
    const reflectBtn = screen.getAllByText('/consciousness/reflect')[0]
    fireEvent.click(reflectBtn)
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_api_explorer.body').length).toBeGreaterThan(0)
    })
    const textarea = document.querySelector('textarea')
    expect(textarea).toBeTruthy()
    if (textarea) {
      fireEvent.change(textarea, { target: { value: '{"text":"hello"}' } })
      expect(textarea).toHaveValue('{"text":"hello"}')
    }
  })
})
