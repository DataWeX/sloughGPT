import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

describe('PhonemeSkeleton', () => {
  it('renders encode skeleton by default', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders score skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="score" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders compare skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="compare" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders practice skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="practice" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders batch skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="batch" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders quiz skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="quiz" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders flashcard skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="flashcard" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders history skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="history" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders synthesize skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="synthesize" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })

  it('renders detect skeleton', async () => {
    const { default: PhonemeSkeleton } = await import('./PhonemeSkeleton')
    render(<PhonemeSkeleton variant="detect" />)
    const skeletons = screen.getAllByRole('presentation')
    expect(skeletons.length).toBeGreaterThan(0)
  })
})
