import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SectionLabel } from './SectionLabel'

describe('SectionLabel', () => {
  it('renders children', () => {
    render(<SectionLabel>Section Title</SectionLabel>)
    expect(screen.getByText('Section Title')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    render(<SectionLabel className="extra">Test</SectionLabel>)
    const el = screen.getByText('Test')
    expect(el.className).toContain('extra')
  })
})
