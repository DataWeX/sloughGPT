/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SearchCategoryBadge } from './SearchCategoryBadge'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
}))

describe('SearchCategoryBadge', () => {
  it('renders the label', () => {
    render(<SearchCategoryBadge type="member" label="Members" />)
    expect(screen.getByText('Members')).toBeDefined()
  })

  it('renders count when provided', () => {
    render(<SearchCategoryBadge type="training" label="Training Jobs" count={5} />)
    expect(screen.getByText('(5)')).toBeDefined()
  })

  it('renders without count when not provided', () => {
    const { container } = render(<SearchCategoryBadge type="dataset" label="Datasets" />)
    expect(container.textContent).toBe('Datasets')
  })

  it('applies default color for known types', () => {
    const { container } = render(<SearchCategoryBadge type="member" label="Members" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-purple-100')
  })

  it('applies custom color class when provided', () => {
    const { container } = render(<SearchCategoryBadge type="custom" label="Custom" colorClass="bg-red-500 text-white" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-red-500')
  })

  it('falls back to muted color for unknown types', () => {
    const { container } = render(<SearchCategoryBadge type="unknown" label="Unknown" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-muted')
  })
})
