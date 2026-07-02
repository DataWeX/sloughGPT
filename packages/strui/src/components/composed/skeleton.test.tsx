import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Skeleton } from './skeleton'

describe('Skeleton', () => {
  it('renders pulse placeholder', () => {
    const html = renderToStaticMarkup(<Skeleton className="h-4 w-full" />)
    expect(html).toContain('animate-pulse')
    expect(html).toContain('rounded-md')
  })
})
