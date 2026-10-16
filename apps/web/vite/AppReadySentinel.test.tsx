/**
 * The invariant the journey suite buys with `data-app-ready`:
 *
 *   at the instant the attribute exists, the current route's content is in
 *   the DOM.
 *
 * Two orders are possible on a real page — the startup stage turns
 * ready/background before the lazy route chunk resolves, or after it — and
 * the sentinel must land the flag after the route content in BOTH.
 */
import { act, render, screen } from '@testing-library/react'
import { lazy, Suspense } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { liveStatusStore, type LiveHealthSnapshot } from '@/hooks/useLiveStatus'

import { AppReadySentinel } from './AppReadySentinel'

function setStage(stage: string) {
  liveStatusStore.setState({ health: { startup_stage: stage } as LiveHealthSnapshot })
}

describe('AppReadySentinel', () => {
  let root: HTMLElement

  beforeEach(() => {
    root = document.createElement('div')
    root.id = 'root'
    document.body.appendChild(root)
    liveStatusStore.setState({ health: null })
  })

  afterEach(() => {
    root.remove()
    liveStatusStore.setState({ health: null })
  })

  it('does not declare readiness before the route chunk commits, even when already booted', async () => {
    // Warm backend: the app is booted before this route's chunk lands.
    setStage('background')

    let resolveChunk: () => void = () => {}
    const chunk = new Promise<void>((resolve) => {
      resolveChunk = resolve
    })
    const Route = lazy(() =>
      chunk.then(() => ({
        default: () => (
          <div>
            <textarea data-testid="composer" />
          </div>
        ),
      })),
    )

    render(
      <Suspense fallback={<div data-testid="fallback" />}>
        <Route />
        <AppReadySentinel />
      </Suspense>,
      { container: root },
    )

    // Suspended boundary: only the fallback is committed, so the flag must
    // still be absent — a test querying now would count 0 inputs.
    expect(screen.getByTestId('fallback')).toBeInTheDocument()
    expect(root.hasAttribute('data-app-ready')).toBe(false)

    await act(async () => {
      resolveChunk()
      await chunk
    })

    // Same commit as the route content: predicate true AND input present.
    expect(root.hasAttribute('data-app-ready')).toBe(true)
    expect(screen.getByTestId('composer')).toBeInTheDocument()
  })

  it('waits for the startup stage, then declares once', () => {
    setStage('critical')
    render(<AppReadySentinel />, { container: root })
    expect(root.hasAttribute('data-app-ready')).toBe(false)

    act(() => {
      setStage('ready')
    })
    expect(root.getAttribute('data-app-ready')).toBe('true')

    // Later stage changes and re-renders never unset or rewrite it.
    act(() => {
      setStage('background')
    })
    expect(root.getAttribute('data-app-ready')).toBe('true')
  })

  it('stays silent while the stage is unknown (no health payload yet)', () => {
    render(<AppReadySentinel />, { container: root })
    expect(root.hasAttribute('data-app-ready')).toBe(false)
  })
})
