import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import { ChatLayout } from './chat-layout'

describe('ChatLayout', () => {
  it('renders header, thread, and composer in order', () => {
    const html = renderToStaticMarkup(
      <ChatLayout
        header={<div data-part="header" />}
        thread={<div data-part="thread" />}
        composer={<div data-part="composer" />}
      />,
    )
    expect(html.indexOf('data-part="header"')).toBeGreaterThan(0)
    expect(html.indexOf('data-part="header"')).toBeLessThan(html.indexOf('data-part="thread"'))
    expect(html.indexOf('data-part="thread"')).toBeLessThan(html.indexOf('data-part="composer"'))
  })

  it('renders thread and composer when header is omitted', () => {
    const html = renderToStaticMarkup(
      <ChatLayout thread={<div data-part="thread" />} composer={<div data-part="composer" />} />,
    )
    expect(html).toContain('data-part="thread"')
    expect(html).toContain('data-part="composer"')
    expect(html).not.toContain('data-part="header"')
  })

  it('passes className to the root column', () => {
    const html = renderToStaticMarkup(
      <ChatLayout thread={<div />} composer={<div />} className="layout-custom" />,
    )
    expect(html).toContain('layout-custom')
  })
})
