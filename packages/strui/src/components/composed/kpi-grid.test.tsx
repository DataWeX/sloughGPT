import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { KpiGrid } from './kpi-grid'

describe('KpiGrid', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<KpiGrid><span>KPI</span></KpiGrid>)
    expect(html).toContain('KPI')
  })

  it('defaults to four columns', () => {
    const html = renderToStaticMarkup(<KpiGrid>x</KpiGrid>)
    expect(html).toContain('grid-cols-2 sm:grid-cols-4')
  })

  it('maps columns=2 to grid-cols-2', () => {
    const html = renderToStaticMarkup(<KpiGrid columns={2}>x</KpiGrid>)
    expect(html).toContain('grid-cols-2')
    expect(html).not.toContain('sm:grid-cols-4')
  })

  it('maps columns=3 to grid-cols-3', () => {
    const html = renderToStaticMarkup(<KpiGrid columns={3}>x</KpiGrid>)
    expect(html).toContain('grid-cols-3')
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(<KpiGrid className="my-grid">x</KpiGrid>)
    expect(html).toContain('my-grid')
  })
})
