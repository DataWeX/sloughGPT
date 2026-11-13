import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { SettingsRow } from './settings-row'

describe('SettingsRow', () => {
  it('renders title', () => {
    const html = renderToStaticMarkup(<SettingsRow title="Dark mode" control={<button>Off</button>} />)
    expect(html).toContain('Dark mode')
  })

  it('renders description when provided', () => {
    const html = renderToStaticMarkup(
      <SettingsRow title="Dark mode" description="Reduce eye strain" control={<button>Off</button>} />,
    )
    expect(html).toContain('Reduce eye strain')
  })

  it('omits description when absent', () => {
    const html = renderToStaticMarkup(<SettingsRow title="Dark mode" control={<button>Off</button>} />)
    expect(html).toContain('Dark mode')
  })

  it('renders control', () => {
    const html = renderToStaticMarkup(<SettingsRow title="Dark mode" control={<button>Off</button>} />)
    expect(html).toContain('<button')
    expect(html).toContain('Off')
  })

  it('passes className to the row', () => {
    const html = renderToStaticMarkup(
      <SettingsRow title="Dark mode" control={<button>Off</button>} className="my-row" />,
    )
    expect(html).toContain('my-row')
  })
})
