import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ReasoningPanel } from './reasoning-panel'

afterEach(() => {
  cleanup()
})

describe('ReasoningPanel', () => {
  it('renders a collapsible details element with the default Reasoning title', () => {
    const html = renderToStaticMarkup(<ReasoningPanel>steps</ReasoningPanel>)
    expect(html).toContain('<details')
    expect(html).toContain('<summary')
    expect(html).toContain('Reasoning')
  })

  it('renders a custom title', () => {
    const html = renderToStaticMarkup(<ReasoningPanel title="Thinking">steps</ReasoningPanel>)
    expect(html).toContain('Thinking')
    expect(html).not.toContain('Reasoning')
  })

  it('renders the children content', () => {
    const html = renderToStaticMarkup(<ReasoningPanel>chain of thought</ReasoningPanel>)
    expect(html).toContain('chain of thought')
  })

  it('is collapsed by default', () => {
    const html = renderToStaticMarkup(<ReasoningPanel>steps</ReasoningPanel>)
    expect(html).not.toContain('open=""')
  })

  it('is open by default when defaultOpen is set', () => {
    const html = renderToStaticMarkup(<ReasoningPanel defaultOpen>steps</ReasoningPanel>)
    expect(html).toContain('open=""')
  })

  it('toggles expansion when the summary is clicked', () => {
    const { container } = render(<ReasoningPanel>steps</ReasoningPanel>)
    const details = container.querySelector('details')
    expect(details).not.toBeNull()
    expect(details?.open).toBe(false)
    fireEvent.click(screen.getByText('Reasoning'))
    expect(details?.open).toBe(true)
    fireEvent.click(screen.getByText('Reasoning'))
    expect(details?.open).toBe(false)
  })

  it('passes className to the details element', () => {
    const html = renderToStaticMarkup(<ReasoningPanel className="panel-custom">steps</ReasoningPanel>)
    expect(html).toContain('panel-custom')
  })
})
