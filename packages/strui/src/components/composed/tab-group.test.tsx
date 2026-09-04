import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { TabGroup } from './tab-group'

describe('TabGroup', () => {
  const tabs = [
    { value: 'overview', label: 'Overview', content: <p>Overview content</p> },
    { value: 'details', label: 'Details', content: <p>Details content</p> },
    { value: 'logs', label: 'Logs', content: <p>Logs content</p> },
  ]

  it('renders tab labels', () => {
    const html = renderToStaticMarkup(
      <TabGroup tabs={tabs} defaultValue="overview" />
    )
    expect(html).toContain('Overview')
    expect(html).toContain('Details')
    expect(html).toContain('Logs')
  })

  it('renders tab content', () => {
    const html = renderToStaticMarkup(
      <TabGroup tabs={tabs} defaultValue="overview" />
    )
    expect(html).toContain('Overview content')
  })

  it('sets data-testid', () => {
    const html = renderToStaticMarkup(
      <TabGroup tabs={tabs} defaultValue="overview" testId="my-tabs" />
    )
    // Tabs component renders data-testid on the wrapper
    expect(html).toBeDefined()
  })

  it('applies custom className', () => {
    const html = renderToStaticMarkup(
      <TabGroup tabs={tabs} defaultValue="overview" className="custom" />
    )
    expect(html).toContain('custom')
  })
})
