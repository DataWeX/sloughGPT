import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Timeline } from './timeline'

describe('Timeline', () => {
  it('renders list of items', () => {
    const html = renderToStaticMarkup(
      <Timeline
        items={[
          { id: '1', title: 'Start', meta: 't0' },
          { id: '2', title: 'Done', meta: 't1' },
        ]}
      />,
    )
    expect(html).toContain('role="list"')
    expect(html).toContain('Start')
    expect(html).toContain('Done')
  })
})
