import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import { SourceList } from './source-list'

describe('SourceList', () => {
  it('renders a Sources block with numbered entries', () => {
    const html = renderToStaticMarkup(
      <SourceList
        sources={[
          { title: 'RAG paper', url: 'https://example.com/rag' },
          { title: 'Local doc', snippet: 'short excerpt' },
        ]}
      />,
    )
    expect(html).toContain('aria-label="Sources"')
    expect(html).toContain('<ol')
    expect(html).toContain('RAG paper')
    expect(html).toContain('Local doc')
  })

  it('links sources that have a url', () => {
    const html = renderToStaticMarkup(
      <SourceList sources={[{ title: 'RAG', url: 'https://example.com/rag' }]} />,
    )
    expect(html).toContain('href="https://example.com/rag"')
    expect(html).toContain('target="_blank"')
    expect(html).toContain('rel="noopener noreferrer"')
  })

  it('renders a plain title for sources without a url', () => {
    const html = renderToStaticMarkup(<SourceList sources={[{ title: 'Local doc' }]} />)
    expect(html).toContain('<span')
    expect(html).toContain('Local doc')
    expect(html).not.toContain('<a ')
  })

  it('renders snippet text when present', () => {
    const html = renderToStaticMarkup(<SourceList sources={[{ title: 'Doc', snippet: 'excerpt' }]} />)
    expect(html).toContain('excerpt')
  })

  it('renders nothing for an empty sources list', () => {
    const html = renderToStaticMarkup(<SourceList sources={[]} />)
    expect(html).toBe('')
  })

  it('passes className to the block', () => {
    const html = renderToStaticMarkup(<SourceList sources={[{ title: 'A' }]} className="sources-custom" />)
    expect(html).toContain('sources-custom')
  })
})
