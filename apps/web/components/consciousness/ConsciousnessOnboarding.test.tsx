import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ConsciousnessOnboarding } from './ConsciousnessOnboarding'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key }),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: vi.fn() }),
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any) => e?.message ?? 'Unknown error',
}))

function clickButton(text: string) {
  const btns = screen.getAllByText(text)
  const btn = btns.find(el => el.tagName === 'BUTTON') ?? btns[0]
  fireEvent.click(btn)
}

describe('ConsciousnessOnboarding', () => {
  const onComplete = vi.fn()
  const onDismiss = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
  })

  it('renders step 1 (welcome)', () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    expect(screen.getByText('consciousness_onboarding.step1_title')).toBeDefined()
  })

  it('navigates to step 2 on Next', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step2_title')).toBeDefined()
    })
  })

  it('PATCHes consciousness config on step 2 → step 3', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step3_title')).toBeDefined()
    })
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/config',
      expect.objectContaining({ method: 'PATCH' })
    )
  })

  it('POSTs preset apply when selected on step 3 → step 4', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step3_title'))
    clickButton('Default')
    clickButton('consciousness_onboarding.next')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step4_title')).toBeDefined()
    })
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/personality/presets/apply',
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('POSTs seed on step 4', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step3_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step4_title'))

    clickButton('consciousness_onboarding.step4_seed')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step4_done')).toBeDefined()
    })
    expect(fetch).toHaveBeenCalledWith(
      'http://localhost:8000/consciousness/seed?count=10',
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('calls onComplete on Finish (step 5)', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step3_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step4_title'))
    clickButton('consciousness_onboarding.step4_skip')
    await waitFor(() => screen.getByText('consciousness_onboarding.step5_title'))

    clickButton('consciousness_onboarding.finish')
    expect(onComplete).toHaveBeenCalled()
  })

  it('calls onDismiss on Skip', () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.skip')
    expect(onDismiss).toHaveBeenCalled()
  })

  it('shows Back button from step 2 onward', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    expect(screen.queryByText('consciousness_onboarding.back')).toBeNull()

    clickButton('consciousness_onboarding.next')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.back')).toBeDefined()
    })
  })

  it('navigates back on Back click', async () => {
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.back')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step1_title')).toBeDefined()
    })
  })

  it('handles fetch error on step 2', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false } as Response)
    render(<ConsciousnessOnboarding onComplete={onComplete} onDismiss={onDismiss} />)
    clickButton('consciousness_onboarding.next')
    await waitFor(() => screen.getByText('consciousness_onboarding.step2_title'))
    clickButton('consciousness_onboarding.next')
    await waitFor(() => {
      expect(screen.getByText('consciousness_onboarding.step2_title')).toBeDefined()
    })
  })
})
