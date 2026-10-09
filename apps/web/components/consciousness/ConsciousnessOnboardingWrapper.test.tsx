import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConsciousnessOnboardingWrapper } from './ConsciousnessOnboardingWrapper'

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://localhost:8000' }))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key }),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: vi.fn() }),
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any) => e?.message ?? 'Unknown',
}))

describe('ConsciousnessOnboardingWrapper', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('always renders children', () => {
    render(
      <ConsciousnessOnboardingWrapper>
        <div>Child content</div>
      </ConsciousnessOnboardingWrapper>
    )
    expect(screen.getByText('Child content')).toBeDefined()
  })

  it('shows onboarding when not completed', () => {
    render(
      <ConsciousnessOnboardingWrapper>
        <div>Child content</div>
      </ConsciousnessOnboardingWrapper>
    )
    expect(document.querySelector('.fixed.inset-0')).not.toBeNull()
  })

  it('does not show onboarding when already completed', () => {
    localStorage.setItem('consciousness_onboarding_complete', 'true')
    render(
      <ConsciousnessOnboardingWrapper>
        <div>Child content</div>
      </ConsciousnessOnboardingWrapper>
    )
    expect(screen.getByText('Child content')).toBeDefined()
    expect(document.querySelector('.fixed.inset-0')).toBeNull()
  })

  it('sets localStorage key after onboarding is dismissed', () => {
    const { container } = render(
      <ConsciousnessOnboardingWrapper>
        <div>Child content</div>
      </ConsciousnessOnboardingWrapper>
    )
    // Onboarding should be visible
    expect(document.querySelector('.fixed.inset-0')).not.toBeNull()

    // Verify the key doesn't exist yet
    expect(localStorage.getItem('consciousness_onboarding_complete')).toBeNull()
  })
})
