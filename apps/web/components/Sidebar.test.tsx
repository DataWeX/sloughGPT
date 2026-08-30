import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('./ThemeSwitcher', () => ({ ThemeSwitcher: () => <div data-testid="theme-switcher" /> }))

vi.mock('@/lib/route-match', () => ({ routeMatchesPath: (p: string, path: string) => p.startsWith(path) }))

import { Sidebar } from './Sidebar'

describe('Sidebar', () => {
  afterEach(cleanup)

  it('renders desktop variant', () => {
    render(<Sidebar />)
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeDefined()
    expect(screen.getByTestId('theme-switcher')).toBeDefined()
  })

  it('renders all nav items', () => {
    render(<Sidebar />)
    expect(screen.getByText('nav.chat')).toBeDefined()
    expect(screen.getByText('nav.training')).toBeDefined()
    expect(screen.getByText('nav.datasets')).toBeDefined()
    expect(screen.getByText('nav.models')).toBeDefined()
    expect(screen.getByText('nav.agents')).toBeDefined()
    expect(screen.getByText('nav.knowledge')).toBeDefined()
    expect(screen.getByText('nav.settings')).toBeDefined()
  })

  it('renders section labels', () => {
    render(<Sidebar />)
    expect(screen.getByText('nav.section.core')).toBeDefined()
    expect(screen.getByText('nav.section.ai')).toBeDefined()
    expect(screen.getByText('nav.section.system')).toBeDefined()
    expect(screen.getByText('nav.section.tools')).toBeDefined()
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

  it('renders app name and console subtitle for desktop', () => {
    render(<Sidebar />)
    expect(screen.getByText('app.name')).toBeDefined()
    expect(screen.getByText('app.console')).toBeDefined()
  })

  it('hides nav labels and section labels when collapsed', () => {
    render(<Sidebar collapsed />)
    expect(screen.queryByText('nav.chat')).toBeNull()
    expect(screen.queryByText('nav.models')).toBeNull()
    expect(screen.queryByText('nav.section.core')).toBeNull()
  })

  it('sets data-collapsed attribute when collapsed', () => {
    const { container } = render(<Sidebar collapsed />)
    const aside = container.querySelector('aside')
    expect(aside?.getAttribute('data-collapsed')).toBe('true')
  })

  it('ignores collapsed when variant is drawer', () => {
    render(<Sidebar variant="drawer" collapsed />)
    expect(screen.getByText('nav.chat')).toBeDefined()
    expect(screen.getByText('nav.section.core')).toBeDefined()
  })
})
