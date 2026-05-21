import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { StepIndicator } from './step-indicator'

describe('StepIndicator', () => {
  it('renders ordered list of steps and highlights current', () => {
    const html = renderToStaticMarkup(
      <StepIndicator steps={['Prepare', 'Train', 'Export']} current={1} />,
    )
    expect(html).toContain('<ol')
    expect(html).toContain('Prepare')
    expect(html).toContain('Train')
    expect(html).toContain('Export')
    expect(html).toContain('border-primary')
  })
})
