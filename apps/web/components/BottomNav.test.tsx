import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { BottomNav } from './BottomNav'

vi.mock('next/link', () => ({
  default: ({ children, href, ...props }: any) => <a href={href} {...props}>{children}</a>,
}))
vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
}))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k.split('.').pop()! }),
}))

afterEach(() => cleanup())

describe('BottomNav', () => {
  it('renders all nav items', () => {
    render(<BottomNav />)
    expect(screen.getAllByText('chat').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('training').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('knowledge').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('models').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('settings').length).toBeGreaterThanOrEqual(1)
  })
  it('has aria-label on nav', () => {
    render(<BottomNav />)
    expect(screen.getAllByLabelText('Bottom navigation').length).toBeGreaterThanOrEqual(1)
  })
  it('sets aria-current on active item', () => {
    render(<BottomNav />)
    const links = screen.getAllByRole('link')
    const active = links.find(l => l.getAttribute('aria-current') === 'page')
    expect(active).toBeDefined()
  })
  it('renders links with correct hrefs', () => {
    render(<BottomNav />)
    const links = screen.getAllByRole('link')
    const hrefs = links.map(l => l.getAttribute('href'))
    expect(hrefs).toContain('/chat')
    expect(hrefs).toContain('/training')
    expect(hrefs).toContain('/knowledge')
    expect(hrefs).toContain('/models')
    expect(hrefs).toContain('/settings')
  })
})
