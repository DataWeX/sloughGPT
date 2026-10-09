/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { ComponentGalleryCard } from './ComponentGalleryCard'

afterEach(() => cleanup())

const defaultProps = {
  switchOn: false,
  onSwitchChange: vi.fn(),
}

describe('ComponentGalleryCard', () => {
  it('renders title', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getByText('Component Gallery')).toBeInTheDocument()
  })

  it('renders button variants', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getByText('Primary')).toBeInTheDocument()
    expect(screen.getAllByText('Secondary').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Destructive').length).toBeGreaterThanOrEqual(1)
  })

  it('renders badge variants', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getAllByText('Default').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Outline').length).toBeGreaterThanOrEqual(1)
  })

  it('renders input placeholder', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Enter text...')).toBeInTheDocument()
  })

  it('renders switch element', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getByLabelText('Gallery switch')).toBeInTheDocument()
  })

  it('renders progress bar', () => {
    render(<ComponentGalleryCard {...defaultProps} />)
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })
})
