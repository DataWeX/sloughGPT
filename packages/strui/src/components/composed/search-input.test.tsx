import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { SearchInput } from './search-input'

describe('SearchInput', () => {
  it('renders search input with default placeholder', () => {
    const html = renderToStaticMarkup(<SearchInput name="q" />)
    expect(html).toContain('type="search"')
    expect(html).toContain('placeholder="Search…"')
    expect(html).toContain('autoComplete="off"')
  })

  it('omits icon padding class when hideIcon', () => {
    const html = renderToStaticMarkup(<SearchInput hideIcon />)
    expect(html).not.toContain('pl-10')
  })
})
