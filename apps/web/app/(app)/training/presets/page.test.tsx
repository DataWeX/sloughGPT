import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import React from 'react'

const mockAddToast = vi.fn()

vi.mock('next/navigation', () => ({
  usePathname: () => '/training/presets',
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
    listTrainingPresets: vi.fn().mockResolvedValue({ presets: [
      { name: 'quick', description: 'Quick training', model: 'gpt2', method: 'finetune', epochs: 1, batch_size: 8, learning_rate: 5e-5, max_seq_length: 512, warmup_steps: 0, weight_decay: 0.01, use_lora: false, lora_rank: 8, lora_alpha: 16, tags: ['fast'] },
    ]}),
    applyTrainingPreset: vi.fn().mockResolvedValue({}),
  },
}))

import TrainingPresetsPage from './page'

describe('TrainingPresetsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially', () => {
    render(<TrainingPresetsPage />)
    const skeletons = document.querySelectorAll('[class*="animate-pulse"]')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders presets after loading', async () => {
    render(<TrainingPresetsPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/Presets/i).length).toBeGreaterThan(0)
    }, { timeout: 5000 })
  })
})
