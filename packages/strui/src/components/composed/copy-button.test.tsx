import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { CopyButton } from './copy-button'

describe('CopyButton', () => {
  it('renders default copy label', () => {
    const html = renderToStaticMarkup(<CopyButton text="hello" />)
    expect(html).toContain('type="button"')
    expect(html).toContain('Copy')
  })

  it('uses custom labels', () => {
    const html = renderToStaticMarkup(
      <CopyButton text="x" labels={['Grab', 'Got it']} />,
    )
    expect(html).toContain('Grab')
  })
})
