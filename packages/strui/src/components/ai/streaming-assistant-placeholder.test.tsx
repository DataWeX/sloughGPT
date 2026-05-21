import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { StreamingAssistantPlaceholder } from './streaming-assistant-placeholder'

describe('StreamingAssistantPlaceholder', () => {
  it('renders aria-hidden skeleton stack', () => {
    const html = renderToStaticMarkup(<StreamingAssistantPlaceholder />)
    expect(html).toContain('aria-hidden')
    expect(html).toContain('animate-pulse')
  })

  it('renders custom line count', () => {
    const html = renderToStaticMarkup(<StreamingAssistantPlaceholder lines={2} />)
    const matches = html.match(/animate-pulse/g)
    expect(matches?.length).toBe(2)
  })
})
