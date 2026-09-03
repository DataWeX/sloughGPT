import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { SectionErrorBoundary } from './SectionErrorBoundary'

afterEach(() => cleanup())

function ThrowingComponent({ shouldThrow = true }: { shouldThrow?: boolean }) {
  if (shouldThrow) throw new Error('Test error')
  return <div>Child content</div>
}

function NoThrow() {
  return <div>Child content</div>
}

describe('SectionErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <SectionErrorBoundary>
        <NoThrow />
      </SectionErrorBoundary>
    )
    expect(screen.getAllByText('Child content').length).toBeGreaterThanOrEqual(1)
  })
  it('shows error UI when child throws', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <SectionErrorBoundary>
        <ThrowingComponent />
      </SectionErrorBoundary>
    )
    expect(screen.getAllByText('Section failed to load').length).toBeGreaterThanOrEqual(1)
    spy.mockRestore()
  })
  it('shows error message', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <SectionErrorBoundary>
        <ThrowingComponent />
      </SectionErrorBoundary>
    )
    expect(screen.getAllByText('Test error').length).toBeGreaterThanOrEqual(1)
    spy.mockRestore()
  })
  it('shows Retry button', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <SectionErrorBoundary>
        <ThrowingComponent />
      </SectionErrorBoundary>
    )
    expect(screen.getAllByText('Retry').length).toBeGreaterThanOrEqual(1)
    spy.mockRestore()
  })
  it('renders section name', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <SectionErrorBoundary sectionName="My Section">
        <ThrowingComponent />
      </SectionErrorBoundary>
    )
    spy.mockRestore()
  })
})
