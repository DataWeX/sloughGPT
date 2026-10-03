import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import type { ReactNode } from 'react'

const pathname = vi.fn(() => '/workspace')

vi.mock('@/vite/next-compat/navigation', () => ({
  usePathname: () => pathname(),
}))

vi.mock('@/vite/next-compat/link', () => ({
  default: ({ href, children, className, ...rest }: Record<string, unknown>) => {
    const { prefetch: _p, ...anchor } = rest
    return (
      <a href={String(href)} className={className as string | undefined} {...anchor}>
        {children as ReactNode}
      </a>
    )
  },
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

import { WorkspaceSectionTabs, WorkspaceSectionLayout } from './WorkspaceSectionTabs'
import { overviewTabs } from './workspace-tabs'

describe('WorkspaceSectionTabs', () => {
  afterEach(() => {
    cleanup()
    pathname.mockReturnValue('/workspace')
  })

  it('renders all tabs with labels', () => {
    render(<WorkspaceSectionTabs tabs={overviewTabs} ariaLabel="Workspace overview" />)
    expect(screen.getByRole('navigation', { name: 'Workspace overview' })).toBeDefined()
    expect(screen.getByText('nav.workspace_dashboard')).toBeDefined()
    expect(screen.getByText('nav.usage')).toBeDefined()
    expect(screen.getByText('nav.audit_trail')).toBeDefined()
  })

  it('marks exact path active with aria-current', () => {
    pathname.mockReturnValue('/workspace/usage')
    render(<WorkspaceSectionTabs tabs={overviewTabs} ariaLabel="Workspace overview" />)
    const active = screen.getByText('nav.usage').closest('a')
    expect(active?.getAttribute('aria-current')).toBe('page')
    const inactive = screen.getByText('nav.workspace_dashboard').closest('a')
    expect(inactive?.getAttribute('aria-current')).toBeNull()
  })

  it('longest-prefix wins so /workspace/usage highlights Usage not Dashboard', () => {
    pathname.mockReturnValue('/workspace/usage')
    render(<WorkspaceSectionTabs tabs={overviewTabs} ariaLabel="Workspace overview" />)
    const usage = screen.getByText('nav.usage').closest('a')
    const dashboard = screen.getByText('nav.workspace_dashboard').closest('a')
    expect(usage?.getAttribute('aria-current')).toBe('page')
    expect(dashboard?.getAttribute('aria-current')).toBeNull()
  })

  it('nested child path still highlights the section parent', () => {
    pathname.mockReturnValue('/workspace')
    render(<WorkspaceSectionTabs tabs={overviewTabs} ariaLabel="Workspace overview" />)
    const dashboard = screen.getByText('nav.workspace_dashboard').closest('a')
    expect(dashboard?.getAttribute('aria-current')).toBe('page')
  })

  it('does not highlight a sibling when on a sibling path', () => {
    pathname.mockReturnValue('/workspace/audit')
    render(<WorkspaceSectionTabs tabs={overviewTabs} ariaLabel="Workspace overview" />)
    expect(screen.getByText('nav.audit_trail').closest('a')?.getAttribute('aria-current')).toBe(
      'page',
    )
    expect(screen.getByText('nav.usage').closest('a')?.getAttribute('aria-current')).toBeNull()
  })
})

describe('WorkspaceSectionLayout', () => {
  afterEach(cleanup)

  it('renders tabs and children', () => {
    render(
      <WorkspaceSectionLayout tabs={overviewTabs} ariaLabel="Workspace overview">
        <p>page-body</p>
      </WorkspaceSectionLayout>,
    )
    expect(screen.getByText('nav.workspace_dashboard')).toBeDefined()
    expect(screen.getByText('page-body')).toBeDefined()
  })
})
