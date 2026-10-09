/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect } from 'vitest'
import { TypographyScaleCard } from './TypographyScaleCard'

afterEach(() => cleanup())

const sampleEntries = [
  { role: 'Page Title', class: 'text-2xl font-semibold', sample: 'Noir Violet', font: 'font-sans', weight: '600' },
  { role: 'Body', class: 'text-sm', sample: 'Warm, sophisticated.', font: 'font-sans', weight: '400' },
] as const

describe('TypographyScaleCard', () => {
  it('renders role labels', () => {
    render(<TypographyScaleCard entries={sampleEntries} />)
    expect(screen.getByText('Page Title')).toBeInTheDocument()
    expect(screen.getByText('Body')).toBeInTheDocument()
  })

  it('renders sample text', () => {
    render(<TypographyScaleCard entries={sampleEntries} />)
    expect(screen.getByText('Noir Violet')).toBeInTheDocument()
    expect(screen.getByText('Warm, sophisticated.')).toBeInTheDocument()
  })

  it('renders CSS class info', () => {
    render(<TypographyScaleCard entries={sampleEntries} />)
    expect(screen.getByText('text-2xl font-semibold')).toBeInTheDocument()
  })

  it('renders multiple entries', () => {
    render(<TypographyScaleCard entries={sampleEntries} />)
    expect(screen.getAllByText(/text-sm|text-2xl/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders entry count', () => {
    render(<TypographyScaleCard entries={sampleEntries} />)
    const roles = screen.getAllByText(/Page Title|Body/)
    expect(roles.length).toBe(2)
  })
})
