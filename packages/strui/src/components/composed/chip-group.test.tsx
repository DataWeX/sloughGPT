import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ChipGroup } from './chip-group'

describe('ChipGroup', () => {
  it('renders chips', () => {
    const html = renderToStaticMarkup(
      <ChipGroup
        chips={[
          { label: 'Python' },
          { label: 'PyTorch' },
          { label: 'GPU' },
        ]}
      />
    )
    expect(html).toContain('Python')
    expect(html).toContain('PyTorch')
    expect(html).toContain('GPU')
  })

  it('renders empty when no chips', () => {
    const html = renderToStaticMarkup(<ChipGroup chips={[]} />)
    expect(html).toContain('flex')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <ChipGroup chips={[{ label: 'A' }]} testId="tags" />
    )
    expect(html).toContain('data-testid="tags"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <ChipGroup chips={[{ label: 'A' }]} className="custom" />
    )
    expect(html).toContain('custom')
  })
})
