import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { PromptComposer } from './prompt-composer'

describe('PromptComposer', () => {
  it('renders textarea and submit control', () => {
    const html = renderToStaticMarkup(
      <PromptComposer value="hi" onValueChange={() => {}} onSubmit={() => {}} />,
    )
    expect(html).toContain('textarea')
    expect(html).toContain('Send')
  })

  it('disables submit when value is empty', () => {
    const html = renderToStaticMarkup(
      <PromptComposer value="   " onValueChange={() => {}} onSubmit={() => {}} />,
    )
    expect(html).toContain('disabled')
  })

  it('omits str-safe-bottom when safeAreaBottom is false', () => {
    const html = renderToStaticMarkup(
      <PromptComposer
        value=""
        onValueChange={() => {}}
        onSubmit={() => {}}
        safeAreaBottom={false}
      />,
    )
    expect(html).not.toContain('str-safe-bottom')
  })

  it('merges textAreaProps data-testid', () => {
    const html = renderToStaticMarkup(
      <PromptComposer
        value=""
        onValueChange={() => {}}
        onSubmit={() => {}}
        textAreaProps={{ 'data-testid': 'composer-input' }}
      />,
    )
    expect(html).toContain('data-testid="composer-input"')
  })
})
