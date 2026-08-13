import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { StatCard } from './stat-card'

describe('StatCard', () => {
  it('renders label and value', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={42} />)
    expect(html).toContain('Requests')
    expect(html).toContain('42')
  })

  it('renders string values', () => {
    const html = renderToStaticMarkup(<StatCard label="Status" value="Online" />)
    expect(html).toContain('Online')
  })

  it('renders icon when provided', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={1} icon={<svg data-testid="icon" />} />)
    expect(html).toContain('<svg')
  })

  it('omits icon when absent', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={1} />)
    expect(html).toContain('Requests')
  })

  it('renders positive trend with plus sign', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={1} trend={{ value: 12.5, positive: true }} />)
    expect(html).toContain('+12.5%')
  })

  it('renders negative trend without plus sign', () => {
    const html = renderToStaticMarkup(<StatCard label="Errors" value={1} trend={{ value: 3, positive: false }} />)
    expect(html).toContain('3%')
    expect(html).not.toContain('+3%')
  })

  it('uses green for positive and red for negative trends', () => {
    const positive = renderToStaticMarkup(<StatCard label="A" value={1} trend={{ value: 1, positive: true }} />)
    const negative = renderToStaticMarkup(<StatCard label="B" value={1} trend={{ value: 1, positive: false }} />)
    expect(positive).toContain('text-green-500')
    expect(negative).toContain('text-red-500')
  })

  it('omits trend when absent', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={1} />)
    expect(html).not.toContain('%')
  })

  it('passes className to the card', () => {
    const html = renderToStaticMarkup(<StatCard label="Requests" value={1} className="my-card" />)
    expect(html).toContain('my-card')
  })
})
