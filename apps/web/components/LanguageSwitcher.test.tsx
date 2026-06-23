// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

const mockSetLocale = vi.fn()

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ locale: 'en', setLocale: mockSetLocale }),
  LOCALES: [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'es', name: 'Español', flag: '🇪🇸' },
    { code: 'ja', name: '日本語', flag: '🇯🇵' },
  ],
}))

import { LanguageSwitcher } from './LanguageSwitcher'

describe('LanguageSwitcher', () => {
  afterEach(cleanup)

  it('renders trigger with current locale code (lowercase, uppercased via CSS)', () => {
    render(<LanguageSwitcher />)
    // JSDOM doesn't apply CSS, so textContent is lowercase 'en'
    const spans = screen.getAllByText('en')
    expect(spans.length).toBeGreaterThanOrEqual(1)
  })

  it('opens dropdown on trigger click', () => {
    render(<LanguageSwitcher />)
    fireEvent.click(screen.getByText('🇬🇧').closest('button')!)
    expect(screen.getByText('🇫🇷 FR')).toBeDefined()
    expect(screen.getByText('🇩🇪 DE')).toBeDefined()
  })

  it('calls setLocale when option clicked', () => {
    render(<LanguageSwitcher />)
    fireEvent.click(screen.getByText('🇬🇧').closest('button')!)
    fireEvent.click(screen.getByText('🇫🇷 FR'))
    expect(mockSetLocale).toHaveBeenCalledWith('fr')
  })
})
