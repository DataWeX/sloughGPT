import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { AppShell } from './app-shell'

describe('AppShell', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<AppShell>content here</AppShell>)
    expect(html).toContain('content here')
  })

  it('renders sidebar when provided', () => {
    const html = renderToStaticMarkup(<AppShell sidebar={<nav>Side</nav>}>main</AppShell>)
    expect(html).toContain('Side')
  })

  it('does not render sidebar wrapper when sidebar is absent', () => {
    const html = renderToStaticMarkup(<AppShell>main</AppShell>)
    expect(html).toContain('main')
  })

  it('renders topBar', () => {
    const html = renderToStaticMarkup(<AppShell topBar={<header>Top</header>}>main</AppShell>)
    expect(html).toContain('Top')
  })

  it('hides sidebar on mobile by default', () => {
    const html = renderToStaticMarkup(<AppShell sidebar={<nav>Side</nav>}>main</AppShell>)
    expect(html).toContain('hidden md:block')
  })

  it('shows sidebar on mobile when showSidebarMobile is set', () => {
    const html = renderToStaticMarkup(
      <AppShell sidebar={<nav>Side</nav>} showSidebarMobile>
        main
      </AppShell>,
    )
    expect(html).toContain('block')
    expect(html).not.toContain('hidden md:block')
  })

  it('passes className to the root element', () => {
    const html = renderToStaticMarkup(<AppShell className="my-shell">main</AppShell>)
    expect(html).toContain('my-shell')
  })
})
