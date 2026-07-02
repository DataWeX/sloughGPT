import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ProgressBar } from './progress-bar'

describe('ProgressBar', () => {
  it('sets progressbar semantics for determinate value', () => {
    const html = renderToStaticMarkup(<ProgressBar value={40} max={100} />)
    expect(html).toContain('role="progressbar"')
    expect(html).toContain('aria-valuenow="40"')
    expect(html).toMatch(/width:\s*40%/)
  })

  it('omits aria-valuenow when indeterminate', () => {
    const html = renderToStaticMarkup(<ProgressBar indeterminate />)
    expect(html).toContain('role="progressbar"')
    expect(html).not.toContain('aria-valuenow')
  })
})
