import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { MessageBubble } from './message-bubble'

describe('MessageBubble', () => {
  it('applies user alignment classes', () => {
    const html = renderToStaticMarkup(<MessageBubble role="user">Hi</MessageBubble>)
    expect(html).toContain('ml-auto')
    expect(html).toContain('Hi')
  })

  it('applies assistant alignment classes', () => {
    const html = renderToStaticMarkup(<MessageBubble role="assistant">Reply</MessageBubble>)
    expect(html).toContain('mr-auto')
  })

  it('applies system tone', () => {
    const html = renderToStaticMarkup(<MessageBubble role="system">Note</MessageBubble>)
    expect(html).toContain('border-dashed')
  })
})
