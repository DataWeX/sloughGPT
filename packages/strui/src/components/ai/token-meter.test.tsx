import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TokenMeter } from './token-meter'

describe('TokenMeter', () => {
  it('formats token count', () => {
    const html = renderToStaticMarkup(<TokenMeter total={500} />)
    expect(html).toContain('500 tok')
  })

  it('shows bar and limit when contextLimit is set', () => {
    const html = renderToStaticMarkup(<TokenMeter total={1000} contextLimit={4000} />)
    expect(html).toContain('1.0k tok')
    expect(html).toMatch(/width:\s*25%/)
    expect(html).toContain('4.0k')
  })
})
