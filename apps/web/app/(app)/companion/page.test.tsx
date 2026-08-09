import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockGetInfo = vi.fn()
const mockListPresets = vi.fn()
const mockGetPrompt = vi.fn()

vi.mock('@/lib/companion-controller', () => ({
  companionController: {
    getInfo: (...args: unknown[]) => mockGetInfo(...args),
    listPresets: (...args: unknown[]) => mockListPresets(...args),
    getPrompt: (...args: unknown[]) => mockGetPrompt(...args),
    chat: (...args: unknown[]) => vi.fn()(...args),
    reset: (...args: unknown[]) => vi.fn()(...args),
    setPersonality: (...args: unknown[]) => vi.fn()(...args),
    setPreset: (...args: unknown[]) => vi.fn()(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import CompanionPage from './page'

describe('CompanionPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetInfo.mockResolvedValue({ traits: null })
    mockListPresets.mockResolvedValue({ presets: [] })
    mockGetPrompt.mockResolvedValue({ system_prompt: '' })
  })

  it('renders page header', async () => {
    render(<CompanionPage />)
    expect(screen.getAllByText('Companion').length).toBeGreaterThanOrEqual(1)
  })

  it('renders preset selector', async () => {
    render(<CompanionPage />)
    await screen.findAllByText(/preset/i)
  })

  it('displays system prompt card', async () => {
    mockGetPrompt.mockResolvedValue({ system_prompt: 'You are a warm companion.' })
    render(<CompanionPage />)
    await screen.findAllByText(/system prompt|warm companion/i)
  })
})
