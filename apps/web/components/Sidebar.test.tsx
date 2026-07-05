import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('./ThemeSwitcher', () => ({ ThemeSwitcher: () => <div data-testid="theme-switcher" /> }))

vi.mock('./CustomDropdown', () => ({ CustomDropdown: ({ trigger }: any) => <div data-testid="custom-dropdown">{trigger}</div> }))

vi.mock('@/lib/route-match', () => ({ routeMatchesPath: (p: string, path: string) => p.startsWith(path) }))

import { Sidebar } from './Sidebar'

describe('Sidebar', () => {
  afterEach(cleanup)

  it('renders desktop variant', () => {
    render(<Sidebar />)
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeDefined()
    expect(screen.getByTestId('theme-switcher')).toBeDefined()
  })

  it('renders workspace nav items', () => {
    render(<Sidebar />)
    expect(screen.getByText('nav.chat')).toBeDefined()
    expect(screen.getByText('nav.training')).toBeDefined()
  })

  it('renders system nav items', () => {
    render(<Sidebar />)
    expect(screen.getByText('nav.models')).toBeDefined()
    expect(screen.getByText('nav.settings')).toBeDefined()
  })

  it('highlights active route', () => {
    render(<Sidebar />)
    const chatLink = screen.getByText('nav.chat').closest('a')
    expect(chatLink?.getAttribute('aria-current')).toBe('page')
  })

  it('renders drawer variant with close button', () => {
    const onClose = vi.fn()
    render(<Sidebar variant="drawer" onClose={onClose} />)
    expect(screen.getByLabelText('sidebar.close')).toBeDefined()
  })

  it('renders account dropdown', () => {
    render(<Sidebar />)
    expect(screen.getByText('sidebar.account')).toBeDefined()
  })

  it('renders app name and console subtitle for desktop', () => {
    render(<Sidebar />)
    expect(screen.getByText('app.name')).toBeDefined()
    expect(screen.getByText('app.console')).toBeDefined()
  })
})
