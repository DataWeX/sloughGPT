import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/shortcuts',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

import ShortcutsPage from './page'

describe('ShortcutsPage', () => {
  it('renders the page title', () => {
    render(<ShortcutsPage />)
    expect(screen.getAllByText(/Keyboard Shortcuts/i).length).toBeGreaterThan(0)
  })

  it('shows global shortcuts', () => {
    render(<ShortcutsPage />)
    expect(screen.getByText('Command palette')).toBeDefined()
    expect(screen.getByText('Toggle dark mode')).toBeDefined()
  })

  it('shows chat shortcuts', () => {
    render(<ShortcutsPage />)
    expect(screen.getByText('New chat')).toBeDefined()
    expect(screen.getByText('Regenerate response')).toBeDefined()
  })

  it('shows category headers', () => {
    render(<ShortcutsPage />)
    expect(screen.getByText('Global')).toBeDefined()
    expect(screen.getByText('Chat')).toBeDefined()
  })
})
