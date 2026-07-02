import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ToolCallCard } from './tool-call-card'

describe('ToolCallCard', () => {
  it('renders name and args preview', () => {
    const html = renderToStaticMarkup(
      <ToolCallCard name="read_file" argsPreview='{"path": "/tmp/x"}' />,
    )
    expect(html).toContain('read_file')
    expect(html).toContain('/tmp/x')
  })

  it('renders state badge', () => {
    const pending = renderToStaticMarkup(<ToolCallCard name="x" state="pending" />)
    expect(pending.toLowerCase()).toContain('pending')
    const err = renderToStaticMarkup(<ToolCallCard name="x" state="error" />)
    expect(err.toLowerCase()).toContain('error')
  })
})
