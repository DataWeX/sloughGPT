import type { ComponentType, ReactNode } from 'react'
import { Component } from 'react'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RouteErrorBoundary, type RouteErrorProps } from './RouteErrorBoundary'

vi.mock('@/lib/dev-log', () => {
  const noop = vi.fn()
  const makeLogger = () => ({
    error: noop,
    warning: noop,
    warn: noop,
    info: noop,
    debug: noop,
    child: makeLogger,
  })
  return { logger: makeLogger() }
})

function TestError({ error, reset }: RouteErrorProps) {
  return (
    <div>
      <p>failed: {error.message}</p>
      <button type="button" onClick={reset}>
        Try again
      </button>
    </div>
  )
}

let shouldThrow = true
function Flaky() {
  if (shouldThrow) throw new Error('boom')
  return <p>recovered</p>
}

function ThrowsRedirect(): never {
  const err = new Error('NEXT_REDIRECT:/login') as Error & { digest?: string }
  err.digest = 'NEXT_REDIRECT;/login'
  throw err
}

function ThrowsNotFound(): never {
  const err = new Error('NEXT_NOT_FOUND') as Error & { digest?: string }
  err.digest = 'NEXT_NOT_FOUND'
  throw err
}

const digestCases: Array<[string, ComponentType]> = [
  ['redirect', ThrowsRedirect],
  ['notFound', ThrowsNotFound],
]

/** Catches whatever RouteErrorBoundary deliberately lets through. */
class SpyBoundary extends Component<{ children: ReactNode }, { caught: boolean }> {
  state = { caught: false }
  static getDerivedStateFromError() {
    return { caught: true }
  }
  render() {
    return this.state.caught ? <p>spy caught</p> : this.props.children
  }
}

describe('RouteErrorBoundary', () => {
  beforeEach(() => {
    shouldThrow = true
  })
  afterEach(() => cleanup())

  it('renders the route error component with error + reset, and reset recovers', () => {
    const { getByText, queryByText } = render(
      <RouteErrorBoundary errorComponent={TestError}>
        <Flaky />
      </RouteErrorBoundary>,
    )
    expect(getByText('failed: boom')).toBeTruthy()

    shouldThrow = false
    fireEvent.click(getByText('Try again'))

    expect(getByText('recovered')).toBeTruthy()
    expect(queryByText('failed: boom')).toBeNull()
  })

  it('falls back to a minimal inline UI when no error component is wired', () => {
    const { getByText } = render(
      <RouteErrorBoundary>
        <Flaky />
      </RouteErrorBoundary>,
    )
    expect(getByText('Something went wrong')).toBeTruthy()
    expect(getByText('boom')).toBeTruthy()
  })

  it.each(digestCases)('lets %s() digests pass through to the outer boundary', (_name, Throws) => {
    const { getByText, queryByText } = render(
      <SpyBoundary>
        <RouteErrorBoundary errorComponent={TestError}>
          <Throws />
        </RouteErrorBoundary>
      </SpyBoundary>,
    )
    expect(getByText('spy caught')).toBeTruthy()
    expect(queryByText(/failed:/)).toBeNull()
  })
})
