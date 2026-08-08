import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { PageSkeleton, CardSkeleton, ListSkeleton } from './PageSkeleton'

describe('PageSkeleton', () => {
  it('renders header by default', () => {
    const { container } = render(<PageSkeleton />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(5) // 2 header + 3 cards
  })

  it('hides header when header=false', () => {
    const { container } = render(<PageSkeleton header={false} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3) // 3 cards only
  })

  it('renders custom card count', () => {
    const { container } = render(<PageSkeleton cards={5} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(7) // 2 header + 5 cards
  })

  it('renders grid layout when grid=true', () => {
    const { container } = render(<PageSkeleton grid />)
    expect(container.querySelector('.grid')).toBeInTheDocument()
  })
})

describe('CardSkeleton', () => {
  it('renders skeleton card', () => {
    const { container } = render(<CardSkeleton />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3)
  })
})

describe('ListSkeleton', () => {
  it('renders default 5 items', () => {
    const { container } = render(<ListSkeleton />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(5)
  })

  it('renders custom item count', () => {
    const { container } = render(<ListSkeleton items={3} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3)
  })
})
