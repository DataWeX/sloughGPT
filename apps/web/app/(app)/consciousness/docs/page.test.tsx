import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessDocsPage from './page'

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
vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers() }))

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessDocsPage />
    </LocaleProvider>
  )
}

describe('ConsciousnessDocsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '{}', headers: new Headers() }))
  })

  it('renders page title', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_docs.page_title').length).toBeGreaterThan(0)
    })
  })

  it('shows API documentation', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_docs.response_example').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint categories', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_docs.cat_core').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_docs.cat_history').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_docs.cat_training').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_docs.cat_personality').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_docs.cat_personas').length).toBeGreaterThan(0)
      expect(screen.getAllByText('consciousness_docs.cat_data').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint method badges', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('GET').length).toBeGreaterThan(0)
      expect(screen.getAllByText('POST').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint paths', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('/consciousness/status').length).toBeGreaterThan(0)
      expect(screen.getAllByText('/consciousness/reflect').length).toBeGreaterThan(0)
    })
  })

  it('shows endpoint descriptions', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('Get consciousness status').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Trigger self-reflection').length).toBeGreaterThan(0)
    })
  })

  it('shows Try It button', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_docs.try_it').length).toBeGreaterThan(0)
    })
  })

  it('shows copy button for response examples', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getAllByText('consciousness_docs.copy').length).toBeGreaterThan(0)
    })
  })
})
