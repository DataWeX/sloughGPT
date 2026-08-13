import { createRef } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ChatThread } from './chat-thread'

afterEach(() => {
  cleanup()
})

describe('ChatThread', () => {
  it('renders its children', () => {
    const html = renderToStaticMarkup(<ChatThread>Hello message</ChatThread>)
    expect(html).toContain('Hello message')
  })

  it('renders a scrollable section with log semantics', () => {
    const html = renderToStaticMarkup(<ChatThread />)
    expect(html).toContain('<section')
    expect(html).toContain('role="log"')
    expect(html).toContain('str-chat-scroll')
  })

  it('announces politely when live', () => {
    const html = renderToStaticMarkup(<ChatThread live />)
    expect(html).toContain('aria-live="polite"')
  })

  it('silences the live region when live is false', () => {
    const html = renderToStaticMarkup(<ChatThread live={false} />)
    expect(html).toContain('aria-live="off"')
  })

  it('uses compact density spacing when density is compact', () => {
    const html = renderToStaticMarkup(<ChatThread density="compact" />)
    expect(html).toContain('gap-2.5')
    expect(html).not.toContain('gap-5')
  })

  it('uses comfortable density spacing by default', () => {
    const html = renderToStaticMarkup(<ChatThread />)
    expect(html).toContain('gap-5')
  })

  it('forwards ref to the section element', () => {
    const ref = createRef<HTMLDivElement>()
    render(<ChatThread ref={ref}>hi</ChatThread>)
    expect(ref.current).not.toBeNull()
    expect(ref.current?.tagName).toBe('SECTION')
  })

  it('passes className to the thread element', () => {
    const html = renderToStaticMarkup(<ChatThread className="thread-custom" />)
    expect(html).toContain('thread-custom')
  })
})
