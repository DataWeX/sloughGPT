import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { ModelStatusPill } from './model-status'

afterEach(() => {
  cleanup()
})

describe('ModelStatusPill', () => {
  it('renders loading label and tone classes', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="loading" />)
    expect(html).toContain('Loading')
    expect(html).toContain('text-primary')
    expect(html).toContain('bg-primary')
  })

  it('renders loaded label and tone classes', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="loaded" />)
    expect(html).toContain('Ready')
    expect(html).toContain('text-green-500')
    expect(html).toContain('bg-green-500')
  })

  it('renders offline label and tone classes', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="offline" />)
    expect(html).toContain('Offline')
    expect(html).toContain('text-red-500')
    expect(html).toContain('bg-red-500')
  })

  it('renders no-model label and tone classes', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="no-model" />)
    expect(html).toContain('No Model')
    expect(html).toContain('text-yellow-500')
    expect(html).toContain('bg-yellow-500')
  })

  it('renders a button', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="offline" />)
    expect(html).toContain('<button')
    expect(html).toContain('type="button"')
  })

  it('shows modelName instead of the status label', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="loaded" modelName="gpt2" />)
    expect(html).toContain('gpt2')
    expect(html).not.toContain('Ready')
  })

  it('applies size classes', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="offline" size="lg" />)
    expect(html).toContain('text-sm')
  })

  it('merges a custom className onto the button', () => {
    const html = renderToStaticMarkup(<ModelStatusPill status="offline" className="my-custom-class" />)
    expect(html).toContain('my-custom-class')
  })

  it('renders a dialog trigger when loaded with model metadata', () => {
    render(<ModelStatusPill status="loaded" modelName="qwen" numParameters={500_000_000} />)
    expect(screen.getAllByRole('button').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('qwen')).toBeTruthy()
  })

  it('fires onClick when clicked', () => {
    const onClick = vi.fn()
    render(<ModelStatusPill status="offline" onClick={onClick} />)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})
