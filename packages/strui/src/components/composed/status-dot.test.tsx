import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { STATUS_DOT_TONE_CLASSES, StatusDot } from './status-dot'

describe('StatusDot', () => {
  it('applies tone classes', () => {
    expect(STATUS_DOT_TONE_CLASSES.success).toBe('bg-success')
    const html = renderToStaticMarkup(<StatusDot tone="success" />)
    expect(html).toContain('bg-success')
  })

  it('renders ping layer when pulse is true', () => {
    const html = renderToStaticMarkup(<StatusDot tone="primary" pulse />)
    expect(html).toContain('animate-ping')
  })

  it('shows label when showLabel is true', () => {
    const html = renderToStaticMarkup(<StatusDot label="Online" showLabel />)
    expect(html).toContain('Online')
  })
})
