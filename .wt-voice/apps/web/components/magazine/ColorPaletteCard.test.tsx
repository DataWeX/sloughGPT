/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect } from 'vitest'
import { ColorPaletteCard } from './ColorPaletteCard'

afterEach(() => cleanup())

const sampleColors = [
  { name: '--primary', rgb: '124 82 196', label: 'Primary', desc: 'Buttons, links' },
  { name: '--background', rgb: '248 246 252', label: 'Background', desc: 'Page background' },
] as const

describe('ColorPaletteCard', () => {
  it('renders title', () => {
    render(<ColorPaletteCard title="Light Mode" colors={sampleColors} />)
    expect(screen.getByText('Light Mode')).toBeInTheDocument()
  })

  it('renders color labels', () => {
    render(<ColorPaletteCard title="Palette" colors={sampleColors} />)
    expect(screen.getByText('Primary')).toBeInTheDocument()
    expect(screen.getByText('Background')).toBeInTheDocument()
  })

  it('renders RGB values', () => {
    render(<ColorPaletteCard title="Palette" colors={sampleColors} />)
    expect(screen.getByText('124 82 196')).toBeInTheDocument()
    expect(screen.getByText('248 246 252')).toBeInTheDocument()
  })

  it('renders descriptions', () => {
    render(<ColorPaletteCard title="Palette" colors={sampleColors} />)
    expect(screen.getByText('Buttons, links')).toBeInTheDocument()
  })

  it('renders swatch elements', () => {
    render(<ColorPaletteCard title="Palette" colors={sampleColors} />)
    const swatches = screen.getAllByRole('img')
    expect(swatches.length).toBe(2)
  })
})
