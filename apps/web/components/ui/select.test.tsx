/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Select } from './select'

afterEach(cleanup)

const options = [
  { value: 'a', label: 'Option A' },
  { value: 'b', label: 'Option B' },
  { value: 'c', label: 'Option C' },
]

describe('Select', () => {
  it('renders trigger with placeholder', () => {
    render(<Select value="" onValueChange={vi.fn()} options={options} placeholder="Pick one" />)
    expect(screen.getByText('Pick one')).toBeInTheDocument()
  })

  it('renders trigger with selected label', () => {
    render(<Select value="b" onValueChange={vi.fn()} options={options} />)
    expect(screen.getByText('Option B')).toBeInTheDocument()
  })

  it('renders trigger with default placeholder', () => {
    render(<Select value="" onValueChange={vi.fn()} options={options} />)
    expect(screen.getByText('Select...')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(
      <Select value="" onValueChange={vi.fn()} options={options} className="custom-class" />
    )
    const button = container.querySelector('button')
    expect(button).toHaveClass('custom-class')
  })
})
