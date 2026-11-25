import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { redirect } from './navigation'
import { RouteDigestBoundary } from './digest-boundary'

function Boom(): null {
  redirect('/monitoring')
  return null
}

function LocationProbe() {
  const loc = useLocation()
  return <div data-testid="path">{loc.pathname}</div>
}

describe('RouteDigestBoundary', () => {
  it('catches next redirect digest and navigates', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <MemoryRouter initialEntries={['/errors']}>
        <RouteDigestBoundary>
          <Routes>
            <Route path="/errors" element={<Boom />} />
            <Route
              path="/monitoring"
              element={
                <>
                  <div>monitoring-page</div>
                  <LocationProbe />
                </>
              }
            />
            <Route path="*" element={<div>fallback</div>} />
          </Routes>
        </RouteDigestBoundary>
      </MemoryRouter>,
    )
    expect(screen.getByTestId('path').textContent).toBe('/monitoring')
    expect(screen.getByText('monitoring-page')).toBeInTheDocument()
  })

  it('captures non-digest crashes into a fallback instead of re-creating the tree', () => {
    // Card 20261123_073036: getDerivedStateFromError returned null for a
    // plain TypeError, so render() re-emitted children, which re-threw on
    // every retry ("React will try to recreate this component tree" ×2322).
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    function Crash(): never {
      throw new TypeError("Cannot destructure property 'sessions' of 'bootstrap.members'")
    }
    render(
      <MemoryRouter initialEntries={['/crash']}>
        <RouteDigestBoundary>
          <Routes>
            <Route path="/crash" element={<Crash />} />
          </Routes>
        </RouteDigestBoundary>
      </MemoryRouter>,
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText(/Cannot destructure/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument()
    // The boundary caught it once — no recreation-loop console spam.
    const recreationSpam = consoleError.mock.calls.filter((args) =>
      String(args[0]).includes('recreate this component tree'),
    )
    expect(recreationSpam).toHaveLength(0)
    consoleError.mockRestore()
  })
})
