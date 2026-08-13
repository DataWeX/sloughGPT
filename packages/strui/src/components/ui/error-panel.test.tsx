import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { useErrorStore } from '../../lib/error-store'
import { ErrorPanel } from './error-panel'

afterEach(() => {
  useErrorStore.setState({ errors: [] })
  cleanup()
})

describe('ErrorPanel', () => {
  it('renders nothing when there are no errors', () => {
    const html = renderToStaticMarkup(<ErrorPanel />)
    expect(html).toBe('')
  })

  it('renders error title and message from the store', () => {
    useErrorStore.getState().addError(new Error('boom'), { title: 'Load failed', severity: 'error', source: 'fetch' })
    render(<ErrorPanel />)
    expect(screen.getByText('Load failed')).toBeTruthy()
    expect(screen.getByText('boom')).toBeTruthy()
  })

  it('shows the issue count in the toggle button', () => {
    useErrorStore.getState().addError(new Error('one'), { severity: 'error' })
    useErrorStore.getState().addError(new Error('two'), { severity: 'error' })
    render(<ErrorPanel />)
    expect(screen.getAllByText(/2 issues/).length).toBeGreaterThan(0)
  })

  it('renders a copy button per error', () => {
    useErrorStore.getState().addError(new Error('x'), { title: 'Copy me' })
    render(<ErrorPanel />)
    expect(screen.getByLabelText('Copy error')).toBeTruthy()
  })

  it('dismisses a single error via the dismiss button', () => {
    useErrorStore.getState().addError(new Error('bad'), { title: 'Boom' })
    render(<ErrorPanel />)
    fireEvent.click(screen.getByLabelText('Dismiss'))
    expect(screen.queryByText('Boom')).toBeNull()
  })

  it('clears all errors via Clear all', () => {
    useErrorStore.getState().addError(new Error('one'), { title: 'One' })
    useErrorStore.getState().addError(new Error('two'), { title: 'Two' })
    render(<ErrorPanel />)
    fireEvent.click(screen.getByText('Clear all'))
    expect(screen.queryByText('One')).toBeNull()
    expect(screen.queryByText('Two')).toBeNull()
  })

  it('styles the toggle red when there are errors', () => {
    useErrorStore.getState().addError(new Error('x'), { severity: 'error' })
    render(<ErrorPanel />)
    const toggle = screen.getAllByText('1 issue').find((el) => el.closest('button'))?.closest('button')
    expect(toggle?.className).toContain('bg-red-500')
  })

  it('styles the toggle orange when there are only warnings', () => {
    useErrorStore.getState().addError(new Error('y'), { severity: 'warning' })
    render(<ErrorPanel />)
    const toggle = screen.getAllByText('1 issue').find((el) => el.closest('button'))?.closest('button')
    expect(toggle?.className).toContain('bg-orange-500')
  })
})
