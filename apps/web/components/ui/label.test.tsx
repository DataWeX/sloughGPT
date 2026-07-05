/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Label } from './label'

afterEach(cleanup)

describe('Label', () => {
  it('renders text content', () => {
    render(<Label>Username</Label>)
    expect(screen.getByText('Username')).toBeInTheDocument()
  })

  it('renders as label element', () => {
    render(<Label>Email</Label>)
    expect(screen.getByText('Email').tagName).toBe('LABEL')
  })

  it('associates with input via htmlFor', () => {
    render(<Label htmlFor="email">Email</Label>)
    expect(screen.getByText('Email')).toHaveAttribute('for', 'email')
  })

  it('applies custom className', () => {
    const { container } = render(<Label className="custom-label">Name</Label>)
    expect(container.firstChild).toHaveClass('custom-label')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Label ref={ref}>Name</Label>)
    expect(ref).toHaveBeenCalled()
  })
})
