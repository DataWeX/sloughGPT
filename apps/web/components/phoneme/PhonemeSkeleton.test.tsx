import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeSkeleton from './PhonemeSkeleton'

describe('PhonemeSkeleton', () => {
  it('renders encode skeleton by default', () => {
    const { container } = render(<PhonemeSkeleton />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders score skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="score" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders compare skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="compare" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders practice skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="practice" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders batch skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="batch" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders quiz skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="quiz" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders flashcard skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="flashcard" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders history skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="history" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders synthesize skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="synthesize" />)
    expect(container.firstChild).toBeTruthy()
  })

  it('renders detect skeleton', () => {
    const { container } = render(<PhonemeSkeleton variant="detect" />)
    expect(container.firstChild).toBeTruthy()
  })
})
