import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { NavRail, NavRailLink } from './nav-rail'

describe('NavRail', () => {
  it('renders a labeled nav', () => {
    const html = renderToStaticMarkup(<NavRail>items</NavRail>)
    expect(html).toContain('<nav')
    expect(html).toContain('aria-label="Main"')
    expect(html).toContain('items')
  })

  it('renders header when provided', () => {
    const html = renderToStaticMarkup(<NavRail header={<a href="/">Home</a>}>items</NavRail>)
    expect(html).toContain('Home')
  })

  it('does not render header block when absent', () => {
    const html = renderToStaticMarkup(<NavRail>items</NavRail>)
    expect(html).toContain('items')
  })

  it('passes className to the nav', () => {
    const html = renderToStaticMarkup(<NavRail className="my-rail">items</NavRail>)
    expect(html).toContain('my-rail')
  })
})

describe('NavRailLink', () => {
  it('renders an anchor with children', () => {
    const html = renderToStaticMarkup(<NavRailLink href="/chat">Chat</NavRailLink>)
    expect(html).toContain('<a')
    expect(html).toContain('href="/chat"')
    expect(html).toContain('Chat')
  })

  it('passes through extra props', () => {
    const html = renderToStaticMarkup(<NavRailLink href="/x" target="_blank">X</NavRailLink>)
    expect(html).toContain('target="_blank"')
  })

  it('adds active styling when active', () => {
    const html = renderToStaticMarkup(<NavRailLink href="/x" active>X</NavRailLink>)
    expect(html).toContain('bg-muted/60')
    expect(html).toContain('border-border')
  })

  it('omits active styling when inactive', () => {
    const html = renderToStaticMarkup(<NavRailLink href="/x">X</NavRailLink>)
    expect(html).not.toContain('bg-muted/60')
  })

  it('passes className to the anchor', () => {
    const html = renderToStaticMarkup(<NavRailLink href="/x" className="my-link">X</NavRailLink>)
    expect(html).toContain('my-link')
  })
})
