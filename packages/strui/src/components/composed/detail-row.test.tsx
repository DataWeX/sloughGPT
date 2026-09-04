import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DetailRow } from './detail-row'

describe('DetailRow', () => {
  it('renders label and value', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="Model" value="GPT-2" />
    )
    expect(html).toContain('Model')
    expect(html).toContain('GPT-2')
  })

  it('applies mono class', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="Path" value="/models/gpt2" mono />
    )
    expect(html).toContain('font-mono')
  })

  it('renders as link when href provided', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="Docs" value="View" href="https://example.com" />
    )
    expect(html).toContain('href="https://example.com"')
    expect(html).toContain('target="_blank"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="A" value="B" className="custom" />
    )
    expect(html).toContain('custom')
  })

  it('applies labelClassName', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="A" value="B" labelClassName="text-red-500" />
    )
    expect(html).toContain('text-red-500')
  })

  it('applies valueClassName', () => {
    const html = renderToStaticMarkup(
      <DetailRow label="A" value="B" valueClassName="text-green-500" />
    )
    expect(html).toContain('text-green-500')
  })
})
