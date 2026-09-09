import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/components/planner/BuffetEngine', () => ({
  BuffetEngine: () => <div data-testid="buffet-engine" />,
}))

import PlannerPage from './page'

describe('PlannerPage', () => {
  afterEach(() => cleanup())

  it('renders BuffetEngine', () => {
    render(<PlannerPage />)
    expect(screen.getByTestId('buffet-engine')).toBeTruthy()
  })
})
