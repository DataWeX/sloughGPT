import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AttachmentChip } from './attachment-chip'

afterEach(() => {
  cleanup()
})

describe('AttachmentChip', () => {
  it('renders the file name', () => {
    const html = renderToStaticMarkup(<AttachmentChip name="report.pdf" />)
    expect(html).toContain('report.pdf')
  })

  it('renders a remove button labeled with the name', () => {
    render(<AttachmentChip name="image.png" onRemove={() => {}} />)
    expect(screen.getByRole('button', { name: 'Remove image.png' })).toBeTruthy()
  })

  it('fires onRemove when the remove button is clicked', () => {
    const onRemove = vi.fn()
    render(<AttachmentChip name="image.png" onRemove={onRemove} />)
    fireEvent.click(screen.getByRole('button', { name: 'Remove image.png' }))
    expect(onRemove).toHaveBeenCalledTimes(1)
  })

  it('renders no remove control when onRemove is omitted', () => {
    const html = renderToStaticMarkup(<AttachmentChip name="report.pdf" />)
    expect(html).not.toContain('Remove')
    expect(html).not.toContain('<button')
    render(<AttachmentChip name="report.pdf" />)
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('passes className to the chip', () => {
    const html = renderToStaticMarkup(<AttachmentChip name="a" className="chip-custom" />)
    expect(html).toContain('chip-custom')
  })
})
