import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { JobStatus } from './job-status'

describe('JobStatus', () => {
  it('renders running label', () => {
    const html = renderToStaticMarkup(<JobStatus status="running" />)
    expect(html.toLowerCase()).toContain('running')
  })

  it('renders failed label for error state', () => {
    const html = renderToStaticMarkup(<JobStatus status="error" />)
    expect(html.toLowerCase()).toContain('failed')
  })

  it('renders success label', () => {
    const html = renderToStaticMarkup(<JobStatus status="success" />)
    expect(html.toLowerCase()).toContain('success')
  })
})
