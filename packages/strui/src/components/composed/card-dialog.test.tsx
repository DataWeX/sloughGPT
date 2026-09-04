import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { CardDialog } from './card-dialog'

describe('CardDialog', () => {
  it('is a valid component with all props', () => {
    const html = renderToStaticMarkup(
      <CardDialog
        open={false}
        onOpenChange={() => {}}
        title="Import Dataset"
        description="Choose a data source"
        size="lg"
        testId="import-dialog"
      >
        <p>Content</p>
      </CardDialog>
    )
    expect(html).toBeDefined()
  })

  it('accepts different sizes', () => {
    const sizes = ['sm', 'md', 'lg', 'xl'] as const
    for (const size of sizes) {
      const html = renderToStaticMarkup(
        <CardDialog
          open={false}
          onOpenChange={() => {}}
          title="Test"
          size={size}
        >
          <p>Content</p>
        </CardDialog>
      )
      expect(html).toBeDefined()
    }
  })
})
