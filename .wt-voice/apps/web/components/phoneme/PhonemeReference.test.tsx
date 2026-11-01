import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeReference from './PhonemeReference'

describe('PhonemeReference', () => {
  it('renders without errors', () => {
    render(<PhonemeReference />)
    expect(screen.getByText('Phoneme Reference')).toBeDefined()
  })

  it('displays total symbol count', () => {
    render(<PhonemeReference />)
    expect(screen.getByText(/\d+ symbols/)).toBeDefined()
  })

  it('renders phoneme groups with examples', () => {
    render(<PhonemeReference />)
    expect(screen.getByText('Stops')).toBeDefined()
    expect(screen.getByText('Vowels & Diphthongs')).toBeDefined()
  })
})
