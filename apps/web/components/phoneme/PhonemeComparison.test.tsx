import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeComparison from './PhonemeComparison'

describe('PhonemeComparison', () => {
  const defaultProps = {
    target: 'hello',
    spoken: 'hello',
    targetPhonemes: ['HH', 'AH', 'L', 'OW'],
    spokenPhonemes: ['HH', 'AH', 'L', 'OW'],
    score: 1.0,
    precision: 1.0,
    recall: 1.0,
  }

  it('renders without errors', () => {
    render(<PhonemeComparison {...defaultProps} />)
    expect(screen.getByText('Pronunciation Analysis')).toBeDefined()
  })

  it('shows perfect score label', () => {
    render(<PhonemeComparison {...defaultProps} />)
    expect(screen.getAllByText('Excellent').length).toBeGreaterThan(0)
  })

  it('shows correct percentage', () => {
    render(<PhonemeComparison {...defaultProps} />)
    expect(screen.getAllByText('100%').length).toBeGreaterThan(0)
  })

  it('shows mismatched phonemes', () => {
    render(
      <PhonemeComparison
        target="hello"
        spoken="jello"
        targetPhonemes={['HH', 'AH', 'L', 'OW']}
        spokenPhonemes={['JH', 'AH', 'L', 'OW']}
        score={0.75}
        precision={0.75}
        recall={0.75}
      />
    )
    expect(screen.getByText('Mismatched Phonemes')).toBeDefined()
  })

  it('shows missing phonemes', () => {
    render(
      <PhonemeComparison
        target="hello"
        spoken="hel"
        targetPhonemes={['HH', 'AH', 'L', 'OW']}
        spokenPhonemes={['HH', 'AH', 'L']}
        score={0.75}
        precision={1.0}
        recall={0.75}
      />
    )
    expect(screen.getByText('Missing:')).toBeDefined()
  })

  it('shows extra phonemes', () => {
    render(
      <PhonemeComparison
        target="hel"
        spoken="hello"
        targetPhonemes={['HH', 'AH', 'L']}
        spokenPhonemes={['HH', 'AH', 'L', 'OW']}
        score={0.75}
        precision={0.75}
        recall={1.0}
      />
    )
    expect(screen.getByText('Extra:')).toBeDefined()
  })
})
