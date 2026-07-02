import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { KeyValueList } from './key-value-list'

describe('KeyValueList', () => {
  it('renders dl with rows', () => {
    const html = renderToStaticMarkup(
      <KeyValueList
        items={[
          { label: 'Run id', value: 'abc', mono: true },
          { label: 'Mode', value: 'train' },
        ]}
      />,
    )
    expect(html).toContain('<dl')
    expect(html).toContain('Run id')
    expect(html).toContain('abc')
    expect(html).toContain('font-mono')
  })
})
