import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
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

  it('renders multiple children', () => {
    render(
      <AppLayoutWrapper>
        <p>first</p>
        <p>second</p>
      </AppLayoutWrapper>,
    )
    expect(screen.getByText('first')).toBeInTheDocument()
    expect(screen.getByText('second')).toBeInTheDocument()
  })

  it('renders nested components', () => {
    render(
      <AppLayoutWrapper>
        <div>
          <span>nested</span>
        </div>
      </AppLayoutWrapper>,
    )
    expect(screen.getByText('nested')).toBeInTheDocument()
  })

  it('renders empty children', () => {
    render(<AppLayoutWrapper>{null}</AppLayoutWrapper>)
    expect(screen.getByTestId('app-layout')).toBeInTheDocument()
  })

  it('renders text children', () => {
    render(<AppLayoutWrapper>text content</AppLayoutWrapper>)
    expect(screen.getByText('text content')).toBeInTheDocument()
  })

  it('renders fragment children', () => {
    render(
      <AppLayoutWrapper>
        <>fragment content</>
      </AppLayoutWrapper>,
    )
    expect(screen.getByText('fragment content')).toBeInTheDocument()
  })
})
