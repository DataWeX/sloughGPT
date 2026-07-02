import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ScrollPanel } from './scroll-panel'

describe('ScrollPanel', () => {
  it('renders bounded scroll region with str-chat-scroll', () => {
    const html = renderToStaticMarkup(
      <ScrollPanel>
        <p>line</p>
      </ScrollPanel>,
    )
    expect(html).toContain('str-chat-scroll')
    expect(html).toContain('overflow-y-auto')
    expect(html).toContain('line')
  })
})
