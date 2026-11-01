import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import React from 'react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/planner',
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k }),
  LOCALES: [],
}))

vi.mock('@/components/planner/BuffetEngine', () => ({
  BuffetEngine: () => <div data-testid="buffet-engine">BuffetEngine</div>,
}))

import PlannerPage from './page'

describe('PlannerPage', () => {
  it('renders the BuffetEngine component', () => {
    render(<PlannerPage />)
    expect(screen.getByTestId('buffet-engine')).toBeDefined()
  })
})
