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
})
