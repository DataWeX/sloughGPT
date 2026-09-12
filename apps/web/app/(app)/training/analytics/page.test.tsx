import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/training/analytics',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    get: vi.fn().mockResolvedValue({ training_analytics: { total_runs: 5, avg_quality: 0.85, convergence_rate: 0.9 } }),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

import TrainingAnalyticsPage from './page'

describe('TrainingAnalyticsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page after loading', async () => {
    render(<TrainingAnalyticsPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/Analytics/i).length).toBeGreaterThan(0)
    }, { timeout: 5000 })
  })
})
