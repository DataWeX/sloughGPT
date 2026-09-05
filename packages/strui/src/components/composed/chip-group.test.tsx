import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ChipGroup } from './chip-group'

describe('ChipGroup', () => {
  it('renders chips', () => {
    const html = renderToStaticMarkup(
      <ChipGroup
        chips={[
          { children: 'Python' },
          { children: 'PyTorch' },
          { children: 'GPU' },
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
      <ChipGroup chips={[{ children: 'A' }]} testId="tags" />
    )
    expect(html).toContain('data-testid="tags"')
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <ChipGroup chips={[{ children: 'A' }]} className="custom" />
    )
    expect(html).toContain('custom')
  })
})
