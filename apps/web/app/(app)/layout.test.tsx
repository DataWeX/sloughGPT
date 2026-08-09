import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import AppLayoutWrapper from './layout'

vi.mock('@/components/AppLayout', () => ({
  default: ({ children }: any) => <div data-testid="app-layout">{children}</div>,
}))

afterEach(() => {
  cleanup()
})

describe('AppLayoutWrapper', () => {
  it('renders children inside AppLayout', () => {
    render(
      <AppLayoutWrapper>
        <p>page content</p>
      </AppLayoutWrapper>,
    )
    expect(screen.getByTestId('app-layout')).toBeInTheDocument()
    expect(screen.getByText('page content')).toBeInTheDocument()
  })
})
